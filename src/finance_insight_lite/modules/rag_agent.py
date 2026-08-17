import os
import re
import time
import threading
import io
from collections import deque
from typing import List, Dict, Any, Optional, Literal
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field, SecretStr
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json

from .query_expansion import QueryExpander
from .hybrid_retriever import HybridRetriever
from .workflow_coordinator import WorkflowCoordinator


# ============================================================================
# Structured Output Schema
# ============================================================================

class DocumentGrade(BaseModel):
    """تقييم مستند واحد ضمن الدفعة"""
    doc_index: int = Field(description="رقم المستند كما ورد في القائمة، يبدأ من 1")
    relevance: Literal["Highly_Relevant", "Moderately_Relevant", "Irrelevant"] = Field(
        description="التصنيف - يجب أن يكون واحداً من هذه القيم الثلاث بالضبط، بدون أي نص إضافي"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="مستوى الثقة بالتصنيف من 0 إلى 1"
    )
    reason_brief: str = Field(
        max_length=120,
        description="سبب مختصر جداً (أقل من 15 كلمة) - يمكن أن يكون بالعربي أو الإنجليزي حسب لغة السؤال"
    )


class BatchGradeResponse(BaseModel):
    """الاستجابة الكاملة لتقييم دفعة المستندات"""
    grades: List[DocumentGrade] = Field(
        description="قائمة تحتوي على تقييم لكل مستند، بنفس عدد المستندات المُدخلة بالضبط"
    )


# ============================================================================
# 0. ADAPTIVE RETRIEVAL DEPTH — Corpus-Aware Adaptive-k
# ============================================================================

class AdaptiveRetrievalDepth:


    def __init__(
        self,
        k_min: int = 3,
        k_upper_bound: int = 20,
        corpus_divisor: int = 15,
        elbow_min_docs: int = 2,
    ):
        self.k_min = k_min
        self.k_upper_bound = k_upper_bound
        self.corpus_divisor = corpus_divisor
        self.elbow_min_docs = elbow_min_docs

    def estimate_corpus_size(self, vector_db) -> Optional[int]:
        try:
            return vector_db._collection.count()
        except Exception:
            pass

        try:
            return vector_db.index.ntotal
        except Exception:
            pass

        try:
            return len(vector_db)
        except Exception:
            pass

        return None

    def compute_k_max(self, vector_db) -> int:
        corpus_size = self.estimate_corpus_size(vector_db)

        if corpus_size is None or corpus_size <= 0:
            return max(self.k_min, min(10, self.k_upper_bound))

        k_max = corpus_size // self.corpus_divisor
        k_max = max(k_max, self.k_min)

        if corpus_size <= 150:
            small_corpus_floor = min(self.k_upper_bound, max(8, int(corpus_size * 0.3)))
            k_max = max(k_max, small_corpus_floor)

        k_max = min(k_max, self.k_upper_bound)
        k_max = min(k_max, corpus_size)
        return int(k_max)

    def detect_elbow(self, scores: List[float]) -> int:
        n = len(scores)
        if n == 0:
            return 0
        if n <= self.elbow_min_docs:
            return n

        scores_arr = np.array(scores, dtype=float)
        diffs = np.abs(np.diff(scores_arr))

        if len(diffs) == 0:
            return n

        elbow = int(np.argmax(diffs)) + 2

        min_safe_docs = max(self.elbow_min_docs, min(5, n))
        elbow = max(elbow, min_safe_docs)

        elbow = min(elbow, n)
        return elbow


# ============================================================================
# 0.5 TPM RATE LIMITER — انتظار محسوب بدل الاصطدام العشوائي بسقف Groq
# ============================================================================

class TPMRateLimiter:


    def __init__(self, tpm_limit: int, safety_margin: float = 0.9, window_seconds: float = 60.0):
        self.tpm_limit = tpm_limit
        self.safety_threshold = tpm_limit * safety_margin
        self.window_seconds = window_seconds
        self._events = deque()  # كل عنصر: [timestamp, estimated_tokens]
        self._lock = threading.Lock()

    @staticmethod
    def estimate_tokens(text: str) -> int:
     
        return max(1, len(text) // 3)

    def _prune(self, now: float):
        while self._events and (now - self._events[0][0]) > self.window_seconds:
            self._events.popleft()

    def wait_if_needed(self, estimated_tokens: int, label: str = ""):
        with self._lock:
            now = time.time()
            self._prune(now)
            used = sum(t for _, t in self._events)

            if used + estimated_tokens > self.safety_threshold and self._events:
                oldest_ts = self._events[0][0]
                sleep_for = self.window_seconds - (now - oldest_ts) + 0.3
                if sleep_for > 0:
                    print(
                        f"⏳ TPM pacing [{label}]: انتظار محسوب {sleep_for:.2f}s "
                        f"(مستخدم تقريباً {used}/{self.tpm_limit} + طلب جديد ~{estimated_tokens})"
                    )
                    time.sleep(sleep_for)
                    now = time.time()
                    self._prune(now)

            self._events.append([now, estimated_tokens])


# ============================================================================
# 1. OPTIMIZED CRAG - Fast Retrieval with Batch Grading
# ============================================================================

class CRAGRetriever:

    MIN_CONFIDENCE_THRESHOLD = 0.45
    GRADING_SNIPPET_CHARS = 600
    TABLE_BYPASS_KEYWORDS = {
        "reconcile", "reconciliation", "ledger", "bank statement", "duplicate",
        "variance", "mismatch", "pipeline", "budget", "actual", "csv", "excel",
        "spreadsheet", "table", "row", "rows", "invoice", "deal",
        "مطابقة", "تسوية", "دفتر", "كشف", "بنك", "مكرر", "تكرار",
        "فرق", "فروقات", "ميزانية", "فعلي", "جدول", "صف", "فاتورة", "صفقة",
    }

    def __init__(
        self,
        vector_db,
        llm,
        adaptive_depth: Optional[AdaptiveRetrievalDepth] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        query_expander: Optional[QueryExpander] = None,
    ):
        self.vector_db = vector_db
        self.llm = llm
        self.adaptive_depth = adaptive_depth or AdaptiveRetrievalDepth()

        self.hybrid_retriever = hybrid_retriever
        self.query_expander = query_expander

        self.structured_llm = self.llm.with_structured_output(BatchGradeResponse)

        self.batch_grader_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Strategic Financial Analyst grading document relevance.

IMPORTANT - LANGUAGE HANDLING:
- The question and documents may be in Arabic, English, or mixed.
- Understand and reason about the content in whichever language it appears.
- However, your OUTPUT is always structured via the provided schema (tool call).
  The `relevance` field must ALWAYS be one of the three fixed English labels
  exactly as given: Highly_Relevant, Moderately_Relevant, Irrelevant.
  Never translate, paraphrase, or invent new labels for this field.
- `reason_brief` may be written in the same language as the question (Arabic
  question -> Arabic reason is fine), but keep it under 15 words.

IMPORTANT - SEMANTIC MATCHING (not literal keyword matching):
- The user's question may be phrased very differently from the document's
  wording — it may use synonyms, paraphrasing, a different level of
  formality, or describe the same concept from another angle (e.g. "أداء
  الشركة المالي بآخر 3 شهور" vs a document that says "إيرادات الربع
  الثالث"). These are the SAME topic and should be graded as if the
  question used the document's own wording.
- Judge relevance based on whether the document's underlying financial
  meaning/topic answers the question — NOT on how many words overlap
  literally between the question and the document text.
- Do not lower your relevance classification or confidence merely because
  the question and document use different terminology for the same concept.

CRITICAL RULES:
- Do NOT write any preamble, explanation, or free text outside the schema.
- Return exactly one grade per document, in the same order given.
- Only classify a document as Irrelevant if its actual subject matter is
  unrelated to what is being asked — not because the wording differs. When
  the topic clearly matches but you are unsure about fine details, prefer
  Moderately_Relevant over Irrelevant.

**Core Analytical Framework (for your reasoning, not for output format):**
1. Historical Performance Analysis - trends in revenue, net income, margins
2. Predictive Signal Detection - links between strategic decisions and outcomes
3. Strategic Context Evaluation - business logic behind financial changes

**Document Relevance Criteria:**
- Highly_Relevant: contains quantitative data or context directly answering the question (regardless of exact wording overlap)
- Moderately_Relevant: partial/supporting context only, or same topic but less directly on-point
- Irrelevant: no real financial substance connected to the question's topic"""),
            ("human", "Question: {question}\n\nDocuments:\n{document}\n\nGrade each document now via the schema.")
        ])

    def get_full_table_documents(self, query: str = "") -> List[Any]:
        full_table_documents = list(getattr(self.vector_db, "_small_table_documents", []) or [])
        if not full_table_documents:
            return []

        source_paths = list(getattr(self.vector_db, "_source_paths", []) or [])
        if not source_paths:
            return full_table_documents

        structured_paths = [
            path for path in source_paths
            if str(path).lower().endswith((".csv", ".xlsx", ".xls"))
        ]
        all_sources_are_small_tables = (
            len(structured_paths) == len(source_paths)
            and len(full_table_documents) == len(structured_paths)
        )
        if all_sources_are_small_tables:
            return full_table_documents

        query_lower = query.lower()
        if any(keyword in query_lower for keyword in self.TABLE_BYPASS_KEYWORDS):
            return full_table_documents

        return []

    @staticmethod
    def _dedupe_docs(docs: list, scores: Optional[list] = None):
        seen = set()
        out_docs, out_scores = [], []
        for i, document in enumerate(docs):
            key = document.page_content.strip()
            if key in seen:
                continue
            seen.add(key)
            out_docs.append(document)
            if scores is not None:
                out_scores.append(scores[i])
        return out_docs, (out_scores if scores is not None else None)

    def batch_grade_documents(self, question: str, documents: List[Any]) -> List[bool]:
        if not documents:
            return []

        docs_text = "\n\n".join([
            f"Document {i + 1} [Page {doc.metadata.get('page')}]:\n{doc.page_content[:self.GRADING_SNIPPET_CHARS]}"
            for i, doc in enumerate(documents)
        ])

        try:
            response: BatchGradeResponse = self.structured_llm.invoke(
                self.batch_grader_prompt.format(
                    question=question,
                    document=docs_text
                )
            )

            grade_map = {g.doc_index: g for g in response.grades}

            relevance_results = []
            for i in range(1, len(documents) + 1):
                grade = grade_map.get(i)

                if grade is None:
                    print(f"⚠️ لا يوجد تقييم للمستند {i}, سيُحجب احترازياً")
                    relevance_results.append(False)
                    continue

                is_relevant = (
                    grade.relevance in ("Highly_Relevant", "Moderately_Relevant")
                    and grade.confidence >= self.MIN_CONFIDENCE_THRESHOLD
                )
                relevance_results.append(is_relevant)

            return relevance_results

        except Exception as e:
            print(f"❌ خطأ في التقييم الدفعي: {e} — سيتم حجب كل المستندات احترازياً")
            return [False] * len(documents)

    def _retrieve_with_scores(self, question: str, k: int):
        try:
            pairs = self.vector_db.similarity_search_with_relevance_scores(question, k=k)
            docs = [p[0] for p in pairs]
            scores = [float(p[1]) for p in pairs]
            return docs, scores
        except Exception:
            pass

        try:
            pairs = self.vector_db.similarity_search_with_score(question, k=k)
            docs = [p[0] for p in pairs]
            raw_scores = [float(p[1]) for p in pairs]
            if len(raw_scores) >= 2 and raw_scores[0] > raw_scores[-1]:
                scores = raw_scores
            else:
                max_val = max(raw_scores) if raw_scores else 1.0
                scores = [max_val - s for s in raw_scores]
            return docs, scores
        except Exception:
            pass

        docs = self.vector_db.similarity_search(question, k=k)
        return docs, None

    def _retrieve_candidates(self, question: str, k_max: int):
        if self.hybrid_retriever is not None:
            if self.query_expander is not None:
                _t = time.time()
                queries = self.query_expander.expand(question)
                print(f"⏱️   ├─ Query Expansion: {time.time() - _t:.2f}s")
            else:
                queries = [question]

            _t = time.time()
            result = self.hybrid_retriever.retrieve_with_scores(queries, k_max=k_max)
            print(f"⏱️   ├─ Hybrid Search (BM25+vector, {len(queries)} صياغة): {time.time() - _t:.2f}s")
            return result

        return self._retrieve_with_scores(question, k=k_max)

    def get_relevant_documents(self, question: str, k: Optional[int] = None) -> List[Dict]:
        k_max = k if k is not None else self.adaptive_depth.compute_k_max(self.vector_db)
        print(f"🔍 نافذة الاسترجاع الأولية (k_max): {k_max}")

        candidate_docs, scores = self._retrieve_candidates(question, k_max=k_max)
        candidate_docs, scores = self._dedupe_docs(candidate_docs, scores)

        if not candidate_docs:
            return []

        if scores is not None and len(scores) == len(candidate_docs):
            cutoff = self.adaptive_depth.detect_elbow(scores)
            print(f"📉 نقطة الانكسار (elbow): أخذ {cutoff} من أصل {len(candidate_docs)} حسب توزيع الدرجات")
            initial_docs = candidate_docs[:cutoff]
        else:
            print("ℹ️ لا يدعم الـ vector store استخراج scores، الاعتماد الكامل على CRAG للفلترة")
            initial_docs = candidate_docs

        _t = time.time()
        relevance_flags = self.batch_grade_documents(question, initial_docs)
        print(f"⏱️   └─ CRAG Grading ({len(initial_docs)} مستند): {time.time() - _t:.2f}s")

        relevant_results = [
            {"document": doc, "relevant": True}
            for doc, is_relevant in zip(initial_docs, relevance_flags)
            if is_relevant
        ]

        print(f"📊 عدد المستندات ذات الصلة: {len(relevant_results)}/{len(initial_docs)}")

        if not relevant_results:
            print("⚠️ لم يُعثر على مستندات ذات صلة، استخدام أفضل 2 كحل احتياطي")
            return [{"document": d, "relevant": True} for d in initial_docs[:2]]

        return relevant_results

# ============================================================================
# 2. Self-RAG Verification + Iterative Self-Refinement
# ============================================================================

class NumberAttribution(BaseModel):

    number_in_answer: str = Field(description="The exact number/figure as it appears in the Answer")
    row_label_in_source: str = Field(
        description="The exact row/item label this number is attached to in the Sources "
                    "(copy it as it appears in the source text). If the number cannot be "
                    "found in the Sources at all, write 'NOT_FOUND_IN_SOURCES'."
    )
    matches_question_intent: bool = Field(
        description="True only if row_label_in_source is actually what the Question is asking "
                    "about — not merely a similarly-worded neighboring row/category/period."
    )
    source_file_claimed: str = Field(
        default="",
        description="The source file the Answer claims this number/fact came from, or empty if no source file is claimed"
    )
    source_file_actual: str = Field(
        default="",
        description="The source file shown in the matching Source chunk for this number/fact, or empty if not found"
    )


class VerificationResult(BaseModel):
    """نتيجة تحقق مبنية على schema - بدون أي parsing عبر regex"""
    number_checks: List[NumberAttribution] = Field(
        default_factory=list,
        description="One entry per distinct number/figure mentioned in the Answer. Must be "
                    "filled BEFORE deciding rating/passed — your rating/passed decision should "
                    "follow logically from these checks, not the other way around."
    )
    rating: int = Field(ge=0, le=10, description="Overall accuracy score from 0 to 10")
    passed: bool = Field(description="True only if numbers are accurate and fully supported by sources")
    missing_refs: List[str] = Field(
        default_factory=list,
        description="Specific page numbers or facts referenced in the answer but not found in sources"
    )
    critical_notes: str = Field(
        default="No issues found.",
        max_length=600,
        description="One or two sentences, written in the TARGET RESPONSE LANGUAGE given in "
                    "the prompt, describing the single most important issue (or the "
                    "equivalent of 'No issues found' if the answer is well supported)"
    )


class ConsistencyCheck(BaseModel):
    contradictions_found: bool = Field(
        description="True if the answer makes two conflicting claims about the same entity"
    )
    contradiction_details: List[str] = Field(
        default_factory=list,
        description="For each contradiction: which entity, and the two conflicting claims made about it, quoted briefly"
    )


class AggregationIntent(BaseModel):
    needs_aggregation: bool = Field(
        description="True if the query requires totals, sums, differences, variances, net/profit, or similar arithmetic"
    )
    aggregation_type: Literal["sum", "difference", "variance_pct", "net", "none"] = Field(
        default="none",
        description="The primary arithmetic operation needed, or none"
    )


class ChartIntent(BaseModel):
    wants_chart: bool = Field(
        description="True if the user is explicitly or implicitly asking to see a visualization/chart/graph of the data, in any language"
    )


def _has_chart_keyword(query: str, keywords: List[str]) -> bool:
    query_lower = query.lower()
    for keyword in keywords:
        keyword_lower = keyword.lower()

        if re.fullmatch(r"[a-z0-9]+", keyword_lower):
            suffix = r"\w*" if keyword_lower.endswith("iz") else ""
            if re.search(rf"\b{re.escape(keyword_lower)}{suffix}\b", query_lower):
                return True
            continue

        if keyword_lower in query_lower:
            return True

    return False


class ChartIntentDetector:
    """
    Detects whether a query asks for a rendered visualization, instead of
    relying only on language-specific keyword matching.
    """

    def __init__(
        self,
        fast_llm,
        rate_limiter: Optional["TPMRateLimiter"] = None,
        fallback_keywords: Optional[List[str]] = None,
    ):
        self.fast_llm = fast_llm
        self.rate_limiter = rate_limiter
        self.fallback_keywords = fallback_keywords or []
        self.structured_llm = fast_llm.with_structured_output(ChartIntent)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Decide if the user's query is asking to see a chart,
graph, plot, or visual representation of data in ANY language, including
Arabic, English, or mixed language.

Mark wants_chart=true for explicit visual requests such as "draw me",
"ارسم لي", "show a chart", "visualize", "قارن لي بيانياً", or requests to
represent the answer as a graph/chart.

Mark wants_chart=false for plain requests to compare, summarize, calculate,
or show numbers when the user has not asked for a visual representation. The
user must clearly want something to look at, not just numbers to read."""),
            ("human", "Query: {query}\n\nDoes this need a chart?")
        ])

    def detect(self, query: str) -> bool:
        if _has_chart_keyword(query, self.fallback_keywords):
            return True

        messages = self.prompt.format_messages(query=query)
        prompt_text = "\n".join(str(message.content) for message in messages)

        if self.rate_limiter is not None:
            self.rate_limiter.wait_if_needed(
                TPMRateLimiter.estimate_tokens(prompt_text),
                label="Chart Intent",
            )

        try:
            result: ChartIntent = self.structured_llm.invoke(messages)
            return bool(result.wants_chart)
        except Exception as e:
            print(f"⚠️ Chart intent detection error: {e} — falling back to keyword check")
            return _has_chart_keyword(query, self.fallback_keywords)


class AggregationIntentDetector:
    """Detects whether deterministic arithmetic should be injected."""

    FALLBACK_KEYWORDS = (
        "total", "sum", "variance", "difference", "net", "profit", "loss",
        "aggregate", "collected", "revenue leak",
        "إجمالي", "اجمالي", "مجموع", "فرق", "فروقات", "تباين", "صافي",
        "ربح", "خسارة", "محصل", "المحصلة",
    )

    def __init__(self, fast_llm, rate_limiter: Optional["TPMRateLimiter"] = None):
        self.fast_llm = fast_llm
        self.rate_limiter = rate_limiter
        self.structured_llm = fast_llm.with_structured_output(AggregationIntent)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Decide if the user query requires arithmetic over
multiple rows of financial data. Mark needs_aggregation=true for totals, sums,
net profit/loss, differences, budget-vs-actual variance, bank statement totals,
collected revenue totals, or similar calculations. Use any language, including
Arabic and English.

If the user only asks for a single already-stated fact, mark false."""),
            ("human", "Query: {query}\n\nClassify the aggregation need.")
        ])

    def detect(self, query: str) -> AggregationIntent:
        messages = self.prompt.format_messages(query=query)
        prompt_text = "\n".join(str(message.content) for message in messages)

        if self.rate_limiter is not None:
            self.rate_limiter.wait_if_needed(
                TPMRateLimiter.estimate_tokens(prompt_text),
                label="Aggregation Intent",
            )

        try:
            return self.structured_llm.invoke(messages)
        except Exception as e:
            print(f"⚠️ Aggregation intent detection error: {e} — falling back to keyword check")
            query_lower = query.lower()
            needs = any(keyword in query_lower for keyword in self.FALLBACK_KEYWORDS)
            agg_type = "none"
            if needs:
                if any(kw in query_lower for kw in ["variance", "تباين", "فروقات"]):
                    agg_type = "variance_pct"
                elif any(kw in query_lower for kw in ["net", "profit", "loss", "صافي", "ربح", "خسارة"]):
                    agg_type = "net"
                elif any(kw in query_lower for kw in ["difference", "فرق"]):
                    agg_type = "difference"
                else:
                    agg_type = "sum"
            return AggregationIntent(needs_aggregation=needs, aggregation_type=agg_type)


class ConsistencyChecker:
    """Structured final-answer contradiction check."""

    def __init__(self, fast_llm, rate_limiter: Optional["TPMRateLimiter"] = None):
        self.fast_llm = fast_llm
        self.rate_limiter = rate_limiter
        self.structured_llm = fast_llm.with_structured_output(ConsistencyCheck)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are checking a financial answer for internal
self-contradictions. Look only for conflicts inside the Answer about the same
entity identifier already present in the Context, such as invoice IDs, deal
IDs, entry IDs, line items, or customer names. Examples: the same invoice is
described as both matched and missing; the same deal is assigned two different
statuses; the same line item is given two incompatible values without
explanation.

Return the structured result only."""),
            ("human", """Question: {query}

Context:
{context}

Answer:
{answer}

Check for contradictions now.""")
        ])

    def check(self, query: str, context: str, answer: str) -> ConsistencyCheck:
        messages = self.prompt.format_messages(query=query, context=context, answer=answer)
        prompt_text = "\n".join(str(message.content) for message in messages)
        if self.rate_limiter is not None:
            self.rate_limiter.wait_if_needed(
                TPMRateLimiter.estimate_tokens(prompt_text),
                label="Consistency Check",
            )

        try:
            return self.structured_llm.invoke(messages)
        except Exception as e:
            print(f"⚠️ Consistency check error: {e} — assuming no contradictions")
            return ConsistencyCheck(contradictions_found=False, contradiction_details=[])


class SelfRAGVerifier:

    VERIFICATION_SNIPPET_CHARS = 6000
    MAX_SOURCES_FOR_VERIFICATION = 8

    def __init__(self, llm, rate_limiter: Optional["TPMRateLimiter"] = None):
        self.llm = llm
        self.rate_limiter = rate_limiter
        self.structured_llm = self.llm.with_structured_output(VerificationResult)

        self.verification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a meticulous financial fact-checker.

Compare the 'Answer' against the 'Source Documents' and grade it strictly via
the provided schema. Do not write any text outside the schema.

### LANGUAGE:
- TARGET RESPONSE LANGUAGE: {answer_language}.
- Write `critical_notes` in the TARGET RESPONSE LANGUAGE. If the target is
  Arabic, write critical_notes fully in Arabic; if English, write it fully
  in English. The language of the Question/Answer/Sources you are grading
  must not override this target.
- All other schema fields (relevance labels, row_label_in_source copied
  verbatim from Sources, etc.) keep their normal format regardless of
  language.

STEP 1 — number_checks (do this FIRST, before deciding rating/passed):
- List every distinct number/figure mentioned in the Answer.
- For each one, find its row/label in the Sources and copy that label
  verbatim into row_label_in_source.
- If the number appears in a VERIFIED_CALCULATIONS block, use
  VERIFIED_CALCULATIONS as the row_label_in_source and treat that computed
  value as authoritative.
- Judge matches_question_intent honestly: it is True only if that row/label
  is actually what the Question asked about. A similarly-worded neighboring
  row/category/period is NOT a match, even if the number itself is real and
  present somewhere in the Sources.
- Fill source_file_actual from the [Source: ...] label on the matching Source
  chunk. If the Answer explicitly claims the number came from a named source
  file (for example "bank statement", "ledger.csv", "budget file"), fill
  source_file_claimed with that claimed source name. If the claim and actual
  source conflict, matches_question_intent must be false.

STEP 2 — rating/passed (derive these FROM step 1, don't decide independently):
- rating >= 7 requires every number_check to have matches_question_intent=true
  and every number to be traceable to the Sources.
- If ANY number_check has matches_question_intent=false, source_file_claimed
  conflicts with source_file_actual, or any figure cannot be found in the
  Sources at all, rating must be below 7 and passed must be false — treat a
  mismatched-row/source number exactly like a fabricated number.
- If a VERIFIED_CALCULATIONS block is present and the Answer gives a covered
  total/sum/variance/net figure that differs from the verified value, rating
  must be below 7 and passed must be false.
- missing_refs must list concrete items, and when the issue is a mismatch
  (real number, wrong row/label), say so explicitly (e.g. "figure X is
  attributed to the wrong row/label in the source table") — not vague
  statements.
- critical_notes must always be written in the TARGET RESPONSE LANGUAGE
  above, since it may be shown directly to the user. Keep it under 40 words."""),
            ("human", """Question: {question}

Answer: {answer}

Sources: {sources}

Grade this answer now via the schema.""")
        ])

    @staticmethod
    def _response_language_for_text(text: str) -> str:
        """Detect whether Arabic or English should be the target language."""
        text = text or ""
        arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
        latin_chars = len(re.findall(r"[A-Za-z]", text))
        if arabic_chars and arabic_chars >= latin_chars:
            return "Arabic"
        return "English"

    @staticmethod
    def _source_names_conflict(claimed_source: str, actual_source: str) -> bool:
        claimed = claimed_source.strip().lower()
        actual = actual_source.strip().lower()
        if not claimed or not actual:
            return False
        if claimed in actual or actual in claimed:
            return False

        claimed_tokens = {
            token for token in re.split(r"[^a-z0-9\u0600-\u06ff]+", claimed)
            if token and token not in {"the", "file", "csv", "xlsx", "xls", "pdf"}
        }
        actual_tokens = {
            token for token in re.split(r"[^a-z0-9\u0600-\u06ff]+", actual)
            if token and token not in {"the", "file", "csv", "xlsx", "xls", "pdf"}
        }
        if claimed_tokens and claimed_tokens.issubset(actual_tokens):
            return False

        aliases = {
            "bank statement": {"bank", "statement"},
            "ledger": {"ledger"},
            "budget": {"budget"},
        }
        for phrase, tokens in aliases.items():
            if phrase in claimed and tokens.issubset(actual_tokens):
                return False

        return True

    def verify_answer(self, question: str, answer: str, sources: List[str],
                       answer_language: Optional[str] = None) -> Dict[str, Any]:
        if answer_language is None:
            answer_language = self._response_language_for_text(question)

        try:
            trimmed_sources = "\n\n".join(
                s[:self.VERIFICATION_SNIPPET_CHARS]
                for s in sources[:self.MAX_SOURCES_FOR_VERIFICATION]
            )
            formatted_prompt = self.verification_prompt.format(
                question=question,
                answer=answer,
                sources=trimmed_sources,
                answer_language=answer_language,
            )

            if self.rate_limiter is not None:
                self.rate_limiter.wait_if_needed(
                    TPMRateLimiter.estimate_tokens(formatted_prompt),
                    label="Verification"
                )

            result: VerificationResult = self.structured_llm.invoke(formatted_prompt)

            rating = result.rating
            passed = result.passed
            missing_refs = list(result.missing_refs)

            for check in result.number_checks:
                claimed_source = str(check.source_file_claimed or "").strip().lower()
                actual_source = str(check.source_file_actual or "").strip().lower()
                source_mismatch = self._source_names_conflict(claimed_source, actual_source)
                is_bad = (
                    not check.matches_question_intent
                    or check.row_label_in_source.strip().upper() == "NOT_FOUND_IN_SOURCES"
                    or source_mismatch
                )
                if is_bad:
                    passed = False
                    rating = min(rating, 4)
                    if source_mismatch:
                        mismatch_note = (
                            f"{check.number_in_answer} claimed from source '{check.source_file_claimed}' "
                            f"but matched source '{check.source_file_actual}'"
                        )
                    else:
                        mismatch_note = (
                            f"{check.number_in_answer} attributed to '{check.row_label_in_source}' "
                            f"which does not match the question's intent"
                        )
                    if mismatch_note not in missing_refs:
                        missing_refs.append(mismatch_note)

            return {
                "rating": rating,
                "passed": passed,
                "missing_refs": missing_refs,
                "notes": result.critical_notes,
                "number_checks": [
                    check.model_dump() if hasattr(check, "model_dump") else check.dict()
                    for check in result.number_checks
                ],
            }
        except Exception as e:
            print(f"⚠️ Verification error: {e}")
            fallback_notes = (
                "تعذّر إجراء التحقق الآلي من هذه الإجابة؛ يُرجى التعامل مع "
                "الأرقام أعلاه بحذر إضافي."
                if answer_language == "Arabic" else
                "Automated verification was unavailable for this response; "
                "treat the figures above with extra caution."
            )
            return {
                "rating": 5,
                "passed": False,
                "missing_refs": [],
                "notes": fallback_notes,
                "number_checks": [],
            }


class SelfRefiningAnswerEngine:

    # ========================================================================
    # ========================================================================

    def __init__(self, llm, verifier_llm=None, max_refinement_attempts: int = 1,
                 pass_threshold: int = 7, rate_limiter: Optional["TPMRateLimiter"] = None,
                 verifier_rate_limiter: Optional["TPMRateLimiter"] = None):
        self.llm = llm
        self.rate_limiter = rate_limiter
        self.verifier = SelfRAGVerifier(verifier_llm or llm, rate_limiter=verifier_rate_limiter)
        self.consistency_checker = ConsistencyChecker(verifier_llm or llm, rate_limiter=verifier_rate_limiter)
        self.max_refinement_attempts = max_refinement_attempts
        self.pass_threshold = pass_threshold

        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Conversational Strategic Financial Advisor.

### LANGUAGE:
- TARGET RESPONSE LANGUAGE: {answer_language}.
- Write the entire final answer in the TARGET RESPONSE LANGUAGE, including
  headings, labels, notes, and fallback statements.
- The UI language, retrieved Context language, and Chat History language must
  not override this target. Use Chat History only for meaning and continuity,
  not as a language-style instruction.

### CONVERSATIONAL LOGIC:
- Use the provided 'Chat History' to understand the context of the current question.
- If the user asks a follow-up (e.g., "Why?"), refer to the previous data extracted.

### GROUNDING RULES (read carefully — this is the most important section):
- Base every answer on the retrieved Context only. Do not infer causal
  relationships unless the Context explicitly states them.
- Do NOT state that one metric causes, explains, drives, or correlates with
  another metric unless the Context explicitly says so in words. Two
  numbers appearing near each other, or both being "high"/"low" in the same
  period, is NOT evidence of a relationship between them — present them as
  separate, independent facts instead.
  - Wrong: "Strengths usage rose +80%, which explains the high ROI score."
  - Right: "Strengths usage rose +80%. Separately, the ROI score was 3.71.
    The report does not establish a direct relationship between these two
    metrics."
- If you are tempted to explain *why* something happened and the Context
  does not state the reason explicitly, say plainly that the evidence is
  insufficient to explain the cause — do not fill the gap with a
  plausible-sounding guess.
- Classify KPIs using standard business definitions, and base any
  explanation on what the KPI actually measures (its methodology) — not on
  assumptions about timing, causality, or unstated context.
- NEVER invent budgets, percentages, currency amounts (SAR/USD/etc.),
  timelines, or staffing/headcount numbers unless that exact figure already
  appears in the Context.
- If the Context contains a VERIFIED_CALCULATIONS block, you MUST use those
  exact computed numbers for any covered total, sum, variance, difference,
  net/profit/loss, or aggregate figure. Do not recompute, round differently,
  flip signs, or alter those values based on raw rows.
- If the Context contains a VERIFIED_ANOMALIES block, treat it as mandatory
  evidence. Mention duplicate/revenue-leak warnings when relevant; never
  silently merge, silently drop, or silently count flagged duplicate invoices
  as normal revenue.
- When Context chunks include [Source: ...] labels from more than one source
  file, attribute each material fact to its correct source file by name
  (for example, "per the bank statement" or "per the ledger") and never state
  that a fact appears in a source it was not retrieved from.

### RESPONSE STRUCTURE — decide based on what the question actually asks:

**Case A — Pure extraction/factual question** (asks for a specific number,
value, count, or fact — e.g. "What is X?", "How many...?", "كم عدد...؟",
"ما قيمة...؟"):
- Answer with ONLY the requested fact(s), stated in one or two clear
  sentences, supported by the number(s) from the Context.
- Do NOT add an "Analysis" section and do NOT add a "Suggestions" section.
  If the question doesn't ask for interpretation or recommendations, don't
  volunteer them.
- If the fact is genuinely not in the Context, say so plainly instead of
  guessing.

**Case B — Analysis / explanation / recommendation question** (asks "why",
"how", "what should we do", asks to compare, evaluate, or advise):
- Structure your answer in three clearly labeled parts (use the same
  language as the question for the labels — Arabic labels shown here,
  mirror in English as Facts / Analysis / Suggestions):

الحقائق:
[Only what is explicitly stated in the Context, with numbers. No
interpretation here.]

التحليل:
[Your interpretation of the facts above, clearly framed as interpretation
(e.g. "قد يشير هذا إلى..." / "This may suggest..."), never stated as if it
were a fact from the report. If evidence is insufficient to support an
interpretation the question is asking for, say so explicitly here instead
of guessing.]

الاقتراحات:
[Only include this part if the question actually asks for advice,
recommendations, or next steps. Give AT MOST one or two suggestions total —
pick the single most impactful one (or two, only if both are clearly
distinct and important); never list more even if the Context could support
several.
Write each suggestion as ONE flowing sentence, not as separate
"التوصية:"/"الدليل:" bullet lines. Let the evidence lead into the
recommendation, using a connector like "بناءً على..." / "based on..."
(e.g. "بناءً على ارتفاع استخدام نقاط القوة بنسبة 80%، يُقترح التركيز على..."
/ "Based on the 80% rise in strengths usage, it may be worth focusing
on..."). Do not use dashes, colons, or labeled sub-fields for this — plain
prose only.
Keep suggestions qualitative/directional (e.g. "consider increasing
investment in coaching quality" — not "invest 15,000 SAR in coaching"),
unless the Context itself already contains the number you're citing.]

### OTHER RULES:
- **Currency**: Always include the currency (e.g., SAR, USD) — but only
  when citing a figure that actually appears in the Context.
- **No raw data**: Never output JSON, code blocks, key-value dumps, or any
  other raw/structured data format anywhere in your answer. Everything must
  be written as natural, conversational prose the end user can read directly.
- **Never copy-paste from Context**: Do not restate, list, or dump the raw
  records/rows from the Context verbatim, not even reformatted with different
  spacing or punctuation. Extract only the specific numbers you need and
  weave them into your prose (e.g. summarize totals, averages, trends — do
  not enumerate every single transaction/row).
- **Row/label precision in tables**: The Context may contain tables with
  several rows whose labels are similarly worded (e.g. close variations of
  the same phrase describing different metrics, periods, or categories).
  Before using any number, double-check it is taken from the row/label that
  actually matches what the Question is asking about — not a neighboring row
  that merely looks similar. If the Question's target is ambiguous between
  two similarly-named rows, briefly note the ambiguity rather than guessing.
{chart_response_instruction}"""),
            ("human", "Chat History: {chat_history}\n\nQuestion: {query}\n\nContext: {context}")
        ])

        self.refine_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Conversational Strategic Financial Advisor revising a
previous answer that failed an accuracy check.

### LANGUAGE:
- TARGET RESPONSE LANGUAGE: {answer_language}.
- Write the corrected answer in the TARGET RESPONSE LANGUAGE, even if the
  previous answer used a different language.
- The previous answer, reviewer critique, Context, and Chat History must not
  override this target language.

### GROUNDING RULES (apply these even while revising):
- Do NOT state that one metric causes, explains, drives, or correlates with
  another metric unless the Context explicitly says so in words. If the
  previous answer did this, remove the causal claim and present the two
  facts separately instead.
- Suggestions are professional opinion, not facts — never invent specific
  numbers, amounts, budgets, currency figures (SAR/USD/etc.), percentages,
  timelines, or staffing/headcount numbers that do not appear in the
  Context. Keep suggestions qualitative unless a number is already present
  in the Context.
- If a VERIFIED_CALCULATIONS block appears in the Context, preserve those
  exact values in the corrected answer. If a VERIFIED_ANOMALIES block appears,
  preserve the mandatory anomaly warning where relevant.
- Preserve correct [Source: ...] attribution. Do not move facts between the
  ledger, bank statement, budget file, pipeline file, or any other source.

### REVISION RULES:
1. You will be given your PREVIOUS ANSWER and a REVIEWER CRITIQUE of it.
2. Fix exactly the issues named in the critique — do not introduce new,
   unrelated changes.
3. Base every number strictly on the provided Context. If the critique
   flags a figure that is genuinely not present in the Context, do not
   invent it — state plainly that it is not available in the reviewed
   documents instead of guessing.
3b. If the critique flags a number as attributed to the wrong row/label
   (a real number pulled from a similarly-worded but different row/category/
   period than what the Question asked about), find and use the number from
   the CORRECT matching row/label in the Context instead — do not just drop
   the number or repeat the same mismatch.
4. Keep the same response structure as the original answer:
   - If the Question is a pure extraction/factual question, keep it to just
     the fact(s) — no "التحليل"/"Analysis" section, no "الاقتراحات"/
     "Suggestions" section.
   - If the Question asks for analysis/explanation/recommendations, keep
     the three labeled parts (الحقائق / التحليل / الاقتراحات, or
     Facts/Analysis/Suggestions in English). Keep الاقتراحات to at most one
     or two suggestions, each written as a single flowing sentence that
     weaves in its supporting evidence using "بناءً على..." / "based on..."
     — not separate "التوصية:"/"الدليل:" labels.
   Do not add a Suggestions section that wasn't warranted by the Question,
   and do not remove one that the Question does call for.
5. Do not restate the critique itself to the user — just produce the
   corrected answer.
5b. Never describe your own edit process (e.g. do not write things like
   "removed the unsupported unit X" or "kept the verified figure Y"). Write
   the answer as if it were the first and only version — natural prose from
   the reader's perspective, with zero meta-commentary about revisions.
6. Never output JSON, code blocks, or any raw/structured data format —
   plain conversational text only.
7. Never copy-paste raw records/rows from the Context verbatim. Summarize
   with your own words; only quote the specific figures needed.
{chart_response_instruction}"""),
            ("human", """Question: {query}

Context: {context}

PREVIOUS ANSWER:
{previous_answer}

REVIEWER CRITIQUE:
- Rating: {rating}/10
- Missing/unsupported items: {missing_refs}
- Notes: {notes}

Provide the corrected answer now.""")
        ])

    @staticmethod
    def _strip_json_artifacts(text: str) -> str:

        if not text:
            return text

        cleaned = text

        cleaned = re.sub(r'```(?:json)?\s*[\s\S]*?```', '', cleaned, flags=re.IGNORECASE)


        smart_quote_map = {
            '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
        }
        for bad, good in smart_quote_map.items():
            cleaned = cleaned.replace(bad, good)

        n = len(cleaned)
        out_chars = []
        i = 0
        while i < n:
            ch = cleaned[i]
            if ch in ('[', '{'):
                closing = ']' if ch == '[' else '}'
                depth = 0
                j = i
                end_idx = None
                while j < n:
                    if cleaned[j] == ch:
                        depth += 1
                    elif cleaned[j] == closing:
                        depth -= 1
                        if depth == 0:
                            end_idx = j
                            break
                    j += 1

                if end_idx is not None:
                    candidate = cleaned[i:end_idx + 1]
                    is_data_dump = False

                    try:
                        json.loads(candidate)
                        is_data_dump = True
                    except (json.JSONDecodeError, ValueError):

                        kv_pattern_count = len(
                            re.findall(r'"[^"\n]{1,60}"\s*:\s*', candidate)
                        )
                        if kv_pattern_count >= 3:
                            is_data_dump = True

                    if is_data_dump:
                        i = end_idx + 1
                        continue


            out_chars.append(ch)
            i += 1

        cleaned = ''.join(out_chars)

        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

        return cleaned if cleaned else text.strip()

    @staticmethod
    def _response_language_for_query(query: str) -> str:
        """Return a stable response language target for the user's question."""
        text = query or ""
        arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
        latin_chars = len(re.findall(r"[A-Za-z]", text))

        if arabic_chars and arabic_chars >= latin_chars:
            return "Arabic"
        return "English"

    @staticmethod
    def _chart_response_instruction(chart_requested: bool) -> str:
        if not chart_requested:
            return ""
        return (
            "\n- **Chart requests**: The system is generating a chart separately "
            "for this question. Do NOT describe how to build a chart in prose, "
            "and do NOT add a Suggestions section telling the user to visualize "
            "the data themselves. Answer the underlying question briefly, as if "
            "the chart will appear directly below your text."
        )

    def _generate_initial(self, query: str, context: str, chat_history: list, chart_requested: bool = False) -> str:
        answer_language = self._response_language_for_query(query)
        formatted = self.answer_prompt.format_messages(
            query=query,
            context=context,
            chat_history=chat_history,
            answer_language=answer_language,
            chart_response_instruction=self._chart_response_instruction(chart_requested),
        )
        if self.rate_limiter is not None:
            combined_text = " ".join(str(m.content) for m in formatted)
            self.rate_limiter.wait_if_needed(
                TPMRateLimiter.estimate_tokens(combined_text),
                label="Answer Generation"
            )
        response = self.llm.invoke(formatted)
        return self._strip_json_artifacts(response.content.strip())

    def _refine(
        self,
        query: str,
        context: str,
        previous_answer: str,
        verification: Dict[str, Any],
        chart_requested: bool = False,
    ) -> str:
        answer_language = self._response_language_for_query(query)
        missing_refs_text = ", ".join(verification.get("missing_refs") or []) or "None specified"
        formatted = self.refine_prompt.format_messages(
            query=query,
            context=context,
            previous_answer=previous_answer,
            answer_language=answer_language,
            rating=verification.get("rating", 0),
            missing_refs=missing_refs_text,
            notes=verification.get("notes", ""),
            chart_response_instruction=self._chart_response_instruction(chart_requested),
        )
        if self.rate_limiter is not None:
            combined_text = " ".join(str(m.content) for m in formatted)
            self.rate_limiter.wait_if_needed(
                TPMRateLimiter.estimate_tokens(combined_text),
                label="Refinement"
            )
        response = self.llm.invoke(formatted)
        return self._strip_json_artifacts(response.content.strip())

    _HAS_DIGIT_RE = re.compile(r'[0-9\u0660-\u0669]')

    @staticmethod
    def _unsupported_not_found_checks(verification: Dict[str, Any]) -> List[Dict[str, Any]]:
        checks = verification.get("number_checks") or []
        return [
            check
            for check in checks
            if str(check.get("row_label_in_source", "")).strip().upper() == "NOT_FOUND_IN_SOURCES"
        ]

    @staticmethod
    def _remove_unverified_numbers(answer: str, verification: Dict[str, Any]) -> str:
        unsupported_checks = SelfRefiningAnswerEngine._unsupported_not_found_checks(verification)
        if not unsupported_checks:
            return answer

        unsupported_numbers = [
            str(check.get("number_in_answer", "")).strip()
            for check in unsupported_checks
            if str(check.get("number_in_answer", "")).strip()
        ]
        if not unsupported_numbers:
            return answer

        kept_lines = []
        for line in answer.splitlines():
            if any(number in line for number in unsupported_numbers):
                continue
            kept_lines.append(line)

        cleaned = "\n".join(kept_lines).strip()
        statements = [
            f"The figure {number} was not found in the provided data."
            for number in unsupported_numbers
        ]
        replacement = "\n".join(statements)

        if cleaned:
            return f"{cleaned}\n\n{replacement}"
        return replacement

    def run(
        self,
        query: str,
        context: str,
        chat_history: list,
        source_texts: List[str],
        chart_requested: bool = False,
        use_self_rag: bool = True,
    ) -> Dict[str, Any]:
        answer_language = self._response_language_for_query(query)

        try:
            _t = time.time()
            answer = self._generate_initial(query, context, chat_history, chart_requested=chart_requested)
            print(f"⏱️   ├─ Initial Answer Generation: {time.time() - _t:.2f}s")
        except Exception as e:
            print(f"⚠️ Answer generation error: {e}")
            failure_answer = (
                "تعذّر إنشاء إجابة في الوقت الحالي. يُرجى المحاولة مرة أخرى بعد قليل."
                if answer_language == "Arabic" else
                "We were unable to generate a response at this time. Please try again shortly."
            )
            failure_notes = (
                "فشلت عملية إنشاء الإجابة." if answer_language == "Arabic" else "Generation failed."
            )
            return {
                "answer": failure_answer,
                "verification": {"rating": 0, "passed": False, "missing_refs": [], "notes": failure_notes},
                "attempts_made": 0,
                "self_refine_converged": False,
            }

        if not use_self_rag:
            print("⏱️   ├─ Verification: skipped (use_self_rag=False)")
            return {
                "answer": self._strip_json_artifacts(answer),
                "verification": None,
                "attempts_made": 0,
                "self_refine_converged": None,
            }

        if not self._HAS_DIGIT_RE.search(answer):
            print("⏱️   ├─ Verification: تخطّي (الإجابة بدون أرقام)")
            skip_notes = (
                "لا توجد أرقام في الإجابة؛ تم تخطي التحقق."
                if answer_language == "Arabic" else
                "No numeric figures in the answer; verification skipped."
            )
            verification = {
                "rating": 8,
                "passed": True,
                "missing_refs": [],
                "notes": skip_notes,
            }
            return {
                "answer": self._strip_json_artifacts(answer),
                "verification": verification,
                "attempts_made": 0,
                "self_refine_converged": True,
            }

        attempts = []
        total_rounds = self.max_refinement_attempts + 1

        for round_idx in range(total_rounds):
            _t = time.time()
            verification = self.verifier.verify_answer(
                query, answer, source_texts, answer_language=answer_language
            )
            print(f"⏱️   ├─ Verification round {round_idx + 1}: {time.time() - _t:.2f}s (rating={verification['rating']}, passed={verification['passed']})")
            attempts.append({"answer": answer, "verification": verification})

            if verification["passed"]:
                break

            is_last_round = (round_idx == total_rounds - 1)
            if is_last_round:
                break

            try:
                _t = time.time()
                answer = self._refine(
                    query,
                    context,
                    answer,
                    verification,
                    chart_requested=chart_requested,
                )
                print(f"⏱️   ├─ Refinement round {round_idx + 1}: {time.time() - _t:.2f}s")
            except Exception as e:
                print(f"⚠️ Refinement error on attempt {round_idx + 1}: {e}")
                break

        best_attempt = max(attempts, key=lambda a: a["verification"]["rating"])
        converged = attempts[-1]["verification"]["passed"]
        best_answer = best_attempt["answer"]
        best_verification = best_attempt["verification"]

        if not converged:
            best_answer = self._remove_unverified_numbers(best_answer, best_verification)

        consistency = self.consistency_checker.check(query, context, best_answer)
        if consistency.contradictions_found:
            details = "; ".join(consistency.contradiction_details) or "Internal contradiction found."
            consistency_verification = {
                "rating": min(best_verification.get("rating", 0), 4),
                "passed": False,
                "missing_refs": [f"Internal contradiction: {details}"],
                "notes": "The answer contains conflicting claims about the same entity.",
            }
            try:
                best_answer = self._refine(
                    query,
                    context,
                    best_answer,
                    consistency_verification,
                    chart_requested=chart_requested,
                )
                best_verification = consistency_verification
                converged = False
            except Exception as e:
                print(f"⚠️ Consistency refinement error: {e}")

        return {
            "answer": self._strip_json_artifacts(best_answer),
            "verification": best_verification,
            "attempts_made": len(attempts),
            "self_refine_converged": converged,
        }


# ============================================================================
# 3. VISUALIZATION TOOL - Financial Data Visualization
# ============================================================================

class FinancialDataExtractor:

    def __init__(self, vector_db, llm, adaptive_depth: Optional[AdaptiveRetrievalDepth] = None):
        self.vector_db = vector_db
        self.llm = llm
        self.adaptive_depth = adaptive_depth or AdaptiveRetrievalDepth(
            k_min=4,
            k_upper_bound=25,
            corpus_divisor=10,
        )

    def extract_data_from_query(
        self,
        query: str,
        k: Optional[int] = None,
        docs: Optional[List[Any]] = None,
    ) -> pd.DataFrame:
        improvement_query = self._is_improvement_percentage_query(query)

        if docs is not None and not improvement_query:
            # Reuse the documents already vetted by CRAG for the main answer,
            # instead of doing an independent (unfiltered) retrieval — this
            # keeps the chart consistent with what the text answer actually
            # found (or didn't find).
            print(f"📊 استخدام {len(docs)} مستند تم فرزها مسبقاً (CRAG) لبناء الرسم البياني")
        else:
            k_max = k if k is not None else self.adaptive_depth.compute_k_max(self.vector_db)
            print(f"📊 نافذة الاسترجاع لاستخراج بيانات الرسم البياني (k_max): {k_max}")

            retrieval_query = "نسبة التحسّن البُعد المقاس" if improvement_query else query
            retrieval_k = k_max
            if improvement_query:
                corpus_size = self.adaptive_depth.estimate_corpus_size(self.vector_db)
                if corpus_size:
                    # A comparison chart needs every occurrence of this structured
                    # field, not just the nearest semantic matches.
                    retrieval_k = min(corpus_size, 100)
            docs = self.vector_db.similarity_search(retrieval_query, k=retrieval_k)

        if not docs:
            return pd.DataFrame()

        # Percentage-improvement rows already have a stable structure in the
        # processed documents. Parse them directly so the chart cannot mix
        # percentages with ratings, raw differences, or unrelated metrics.
        if improvement_query:
            improvement_data = self._extract_improvement_percentages(docs)
            if improvement_data:
                return pd.DataFrame(improvement_data)

        combined_text = "\n\n".join([
            f"[Page {doc.metadata.get('page')} | Sheet: {doc.metadata.get('sheet_name', 'N/A')}]\n{doc.page_content}"
            for doc in docs
        ])

        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract structured financial data that DIRECTLY answers the query's
            specific topic/metric.
            RULES:
            1. Only extract items whose label/metric matches what the query is actually
               asking about (e.g. if the query asks about "revenue"/"الإيرادات", only
               extract rows that are actually revenue figures — not unrelated metrics
               like costs, ratings, headcount, or KPIs that merely appear in the same
               context).
            2. Each object MUST have: "label", "value", "currency", "suggestion".
            3. "value" must be a CLEAN number.
            4. If the Context does not contain data that actually matches the query's
               requested metric/topic, return an empty list [] — do NOT substitute a
               different metric just because it is present in the Context.
            5. Every returned row must represent the SAME metric and unit; never mix
               percentages, ratings, differences, and monetary amounts in one result.
            6. STRICT: NO markdown, ONLY JSON array."""),
            ("human", "Query: {query}\n\nContext:\n{combined_text}\n\nJSON:")
        ])

        try:
            response = self.llm.invoke(extraction_prompt.format_messages(
                query=query,
                combined_text=combined_text
            ))
            raw = response.content.strip()

            raw = re.sub(r'```(?:json)?\s*', '', raw)
            raw = raw.replace('```', '').strip()

            data = json.loads(raw)

            if isinstance(data, list):
                valid_data = []
                for item in data:
                    if not isinstance(item, dict): continue

                    item_clean = {str(k).lower().strip(): v for k, v in item.items()}

                    label = item_clean.get('label') or item_clean.get('name') or item_clean.get('description')
                    value = item_clean.get('value') or item_clean.get('amount')
                    currency = item_clean.get('currency', 'SAR')
                    suggestion = item_clean.get('suggestion', 'No specific advice')

                    if label and value is not None:
                        try:
                            clean_val = float(str(value).replace(',', '').replace('$', '').strip())
                            valid_data.append({
                                'label': str(label),
                                'value': clean_val,
                                'currency': str(currency),
                                'suggestion': str(suggestion)
                            })
                        except: continue

                if valid_data:
                    return pd.DataFrame(valid_data).drop_duplicates(subset=['label'])

        except Exception as e:
            print(f"❌ Detailed Debug: Error Type: {type(e).__name__}, Message: {str(e)}")

        print("📌 Falling back to regex extraction...")
        return pd.DataFrame(columns=['label', 'value', 'currency', 'suggestion'])

    @staticmethod
    def _normalize_arabic(text: str) -> str:
        diacritics = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
        return diacritics.sub("", text).replace("ـ", "")

    @classmethod
    def _is_improvement_percentage_query(cls, query: str) -> bool:
        normalized = cls._normalize_arabic(query.lower())
        return any(phrase in normalized for phrase in [
            "نسبة التحسن",
            "نسب التحسن",
            "معدل التحسن",
            "improvement rate",
            "improvement percentage",
        ])

    @staticmethod
    def _extract_improvement_percentages(docs: List[Any]) -> List[Dict[str, Any]]:
        label_pattern = re.compile(
            r"(?:البُعد\s+المقاس|measured\s+dimension|dimension)\s*:\s*([^|\n]+)",
            re.IGNORECASE,
        )
        value_pattern = re.compile(
            r"(?:نسبة\s+التحس[\u064B-\u065F]*ن|improvement\s+(?:rate|percentage))"
            r"\s*:\s*\+?(-?[\d,]+(?:\.\d+)?)\s*%",
            re.IGNORECASE,
        )
        rows = []
        seen_labels = set()

        for doc in docs:
            text = doc.page_content
            label_match = label_pattern.search(text)
            value_match = value_pattern.search(text)
            if not label_match or not value_match:
                continue

            label = label_match.group(1).strip()
            if label in seen_labels:
                continue

            seen_labels.add(label)
            rows.append({
                "label": label,
                "value": float(value_match.group(1).replace(",", "")),
                "currency": "%",
                "suggestion": "",
            })

        return rows


class ChartDecision(BaseModel):
    chart_type: Literal["bar", "line", "pie", "area"] = Field(
        description="The single best chart type for this query and data"
    )
    reason: str = Field(max_length=150, description="One short reason for the choice")


class ChartTypeSelector:
    """
    Decides the best chart type using both the user's question and the shape
    of the actually extracted data.
    """

    def __init__(self, fast_llm, rate_limiter: Optional["TPMRateLimiter"] = None):
        self.fast_llm = fast_llm
        self.rate_limiter = rate_limiter
        self.structured_llm = fast_llm.with_structured_output(ChartDecision)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You choose the best chart type for a financial data
visualization. The question may be in Arabic or English.

Choose exactly one of: bar, line, pie, area.

Guidance:
- line: the data represents a value changing over an ordered time axis
  (months, quarters, years, dates) and the question cares about the trend/
  trajectory over that time.
- area: like line, but the question emphasizes cumulative volume or magnitude
  over time rather than the trend shape itself.
- pie: the data is a small number of categories (roughly 2–6) that are parts
  of one meaningful whole, and the question asks about share/proportion/
  breakdown/percentage of total.
- bar: the default for comparing a metric across discrete categories
  (departments, deals, invoices, line items) with no time ordering and no
  whole-to-part relationship; this is the safe default when nothing else
  clearly fits.

Scatter plots are intentionally unavailable for now because the extracted data
schema has one numeric value per label, not two independent numeric axes.

Base your decision on BOTH the question's intent and the actual data preview
given below. If the labels are clearly categorical names (not dates/periods)
even though the question uses a word like "trend", data shape wins."""),
            ("human", """Question: {query}

Data preview (label -> value):
{data_preview}

Number of data points: {n_points}

Choose the chart type now.""")
        ])

    def select(self, query: str, df: "pd.DataFrame") -> ChartDecision:
        preview_rows = []
        for _, row in df.head(50).iterrows():
            label = row.get("label", "")
            value = row.get("value", "")
            preview_rows.append(f"- {label}: {value}")
        preview = "\n".join(preview_rows)
        messages = self.prompt.format_messages(
            query=query,
            data_preview=preview,
            n_points=len(df),
        )
        prompt_text = "\n".join(str(message.content) for message in messages)

        if self.rate_limiter is not None:
            self.rate_limiter.wait_if_needed(
                TPMRateLimiter.estimate_tokens(prompt_text),
                label="Chart Type Selection",
            )

        try:
            return self.structured_llm.invoke(messages)
        except Exception as e:
            print(f"⚠️ Chart type selection error: {e} — falling back to bar")
            return ChartDecision(
                chart_type="bar",
                reason="selection_failed_safe_default",
            )


class ChartGenerator:
    """Generate Plotly charts with financial styling"""

    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    @staticmethod
    def create_line_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        fig = px.line(df, x=x, y=y, title=title, markers=True,
                      template="plotly_white", color_discrete_sequence=ChartGenerator.COLORS)
        fig.update_layout(
            hovermode='x unified', height=500,
            title_font_size=20,
            xaxis_title={
                "text": x.replace('_', ' ').title(),
                "font": {"size": 14},
            },
            yaxis_title={
                "text": y.replace('_', ' ').title(),
                "font": {"size": 14},
            },
        )
        return fig

    @staticmethod
    def create_bar_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        fig = px.bar(df, x=x, y=y, title=title, template="plotly_white",
                     color_discrete_sequence=ChartGenerator.COLORS)
        fig.update_layout(
            height=500, title_font_size=20,
            xaxis_title=x.replace('_', ' ').title(),
            yaxis_title=y.replace('_', ' ').title()
        )
        fig.update_traces(marker_color=ChartGenerator.COLORS[0])
        return fig

    @staticmethod
    def create_pie_chart(df: pd.DataFrame, names: str, values: str, title: str) -> go.Figure:
        fig = px.pie(df, names=names, values=values, title=title,
                     template="plotly_white", color_discrete_sequence=ChartGenerator.COLORS)
        fig.update_traces(textposition='inside', textinfo='percent')
        fig.update_layout(height=500, title_font_size=20)
        return fig

    @staticmethod
    def create_scatter_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        try:
            df = df.copy()
            if not pd.api.types.is_numeric_dtype(df[x]) and not pd.api.types.is_datetime64_any_dtype(df[x]):
                df["_x_index"] = range(len(df))
                x_plot = "_x_index"
                tickvals = df["_x_index"].tolist()
                ticktext = df[x].astype(str).tolist()
                fig = px.scatter(df, x=x_plot, y=y, title=title, template="plotly_white",
                                color_discrete_sequence=ChartGenerator.COLORS)
                fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
            else:
                fig = px.scatter(df, x=x, y=y, title=title, template="plotly_white",
                                trendline="ols", color_discrete_sequence=ChartGenerator.COLORS)
            fig.update_layout(height=500, title_font_size=20)
            return fig
        except Exception as e:
            print(f"⚠️ Error creating scatter chart: {e}")
            raise

    @staticmethod
    def create_area_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        fig = px.area(df, x=x, y=y, title=title, template="plotly_white",
                      color_discrete_sequence=ChartGenerator.COLORS)
        fig.update_layout(height=500, title_font_size=20)
        return fig


class DeterministicContextAnalyzer:
    """Computes arithmetic and tabular anomaly facts from retrieved rows."""

    MONEY_COLUMNS = ("amount_sar", "budget_sar", "actual_sar")

    def __init__(self, documents: List[Any]):
        self.documents = documents
        self.frames = self._documents_to_frames(documents)

    @staticmethod
    def _source_file(doc) -> str:
        metadata = doc.metadata or {}
        return str(metadata.get("source_file") or metadata.get("source") or "unknown")

    @staticmethod
    def _clean_value(value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return value
        text = str(value).strip()
        if text == "" or text.lower() in {"nan", "none", "null"}:
            return None
        numeric_candidate = text.replace(",", "").replace("SAR", "").replace("sar", "").strip()
        try:
            if re.fullmatch(r"-?\d+(?:\.\d+)?", numeric_candidate):
                number = float(numeric_candidate)
                return int(number) if number.is_integer() else number
        except Exception:
            pass
        return text

    @classmethod
    def _parse_row_sentence(cls, text: str) -> Optional[Dict[str, Any]]:
        row_text = text.splitlines()[-1] if "\n" in text else text
        if " | " not in row_text or ":" not in row_text:
            return None

        row: Dict[str, Any] = {}
        for part in row_text.split(" | "):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            key = key.strip()
            if not key:
                continue
            row[key] = cls._clean_value(value)
        return row or None

    @classmethod
    def _parse_full_csv_blocks(cls, text: str) -> List[pd.DataFrame]:
        frames: List[pd.DataFrame] = []
        for match in re.finditer(r"CSV:\n(?P<csv>[\s\S]*?)(?=\n\nSource file:|\Z)", text):
            csv_text = match.group("csv").strip()
            if not csv_text:
                continue
            try:
                frames.append(pd.read_csv(io.StringIO(csv_text)))
            except Exception:
                continue
        return frames

    @classmethod
    def _documents_to_frames(cls, documents: List[Any]) -> List[tuple[str, pd.DataFrame]]:
        grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
        frames: List[tuple[str, pd.DataFrame]] = []

        for doc in documents:
            source_file = cls._source_file(doc)
            metadata = doc.metadata or {}
            text = doc.page_content or ""

            if metadata.get("table_full_context") or "CSV:\n" in text:
                for frame in cls._parse_full_csv_blocks(text):
                    if not frame.empty:
                        frame = frame.copy()
                        frame["__source_file"] = source_file
                        frames.append((source_file, frame))

            row = cls._parse_row_sentence(text)
            if row:
                row["__source_file"] = source_file
                grouped_rows.setdefault(source_file, []).append(row)

        for source_file, rows in grouped_rows.items():
            if rows:
                frames.append((source_file, pd.DataFrame(rows)))

        return frames

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy()
        normalized.columns = [str(col).strip() for col in normalized.columns]
        for col in DeterministicContextAnalyzer.MONEY_COLUMNS:
            if col in normalized.columns:
                normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
        return normalized

    @staticmethod
    def _format_sar(value: float, signed: bool = False) -> str:
        sign = "+" if signed and value > 0 else ""
        if float(value).is_integer():
            return f"{sign}{int(value):,} SAR"
        return f"{sign}{value:,.2f} SAR"

    def build_verified_context_block(self, query: str, aggregation_intent: Optional[AggregationIntent]) -> str:
        sections: List[str] = []

        calculations = self._build_calculations(query, aggregation_intent)
        if calculations:
            sections.append(
                "VERIFIED_CALCULATIONS (computed by code; use these exact numbers, do not recompute or alter them):\n"
                + "\n".join(f"- {line}" for line in calculations)
            )

        anomalies = self._build_anomalies(query)
        if anomalies:
            sections.append(
                "VERIFIED_ANOMALIES (mandatory to mention when relevant; do not auto-merge or drop these rows):\n"
                + "\n".join(f"- {line}" for line in anomalies)
            )

        return "\n\n".join(sections)

    def _build_calculations(self, query: str, aggregation_intent: Optional[AggregationIntent]) -> List[str]:
        query_lower = FinancialDataExtractor._normalize_arabic(query.lower())
        explicit_need = bool(aggregation_intent and aggregation_intent.needs_aggregation)
        if not explicit_need and not any(
            keyword in query_lower
            for keyword in AggregationIntentDetector.FALLBACK_KEYWORDS
        ):
            return []

        lines: List[str] = []
        for source_file, frame in self.frames:
            df = self._normalize_columns(frame)

            if {"type", "budget_sar", "actual_sar"}.issubset(df.columns):
                type_series = df["type"].astype(str).str.lower()
                expenses = df[type_series == "expense"]
                revenue = df[type_series == "revenue"]
                if not expenses.empty:
                    budget_expenses = float(expenses["budget_sar"].sum())
                    actual_expenses = float(expenses["actual_sar"].sum())
                    lines.extend([
                        f"Total budgeted expenses ({source_file}): {self._format_sar(budget_expenses)}",
                        f"Total actual expenses ({source_file}): {self._format_sar(actual_expenses)}",
                    ])
                if not revenue.empty:
                    budget_revenue = float(revenue["budget_sar"].sum())
                    actual_revenue = float(revenue["actual_sar"].sum())
                    lines.extend([
                        f"Total budgeted revenue ({source_file}): {self._format_sar(budget_revenue)}",
                        f"Total actual revenue ({source_file}): {self._format_sar(actual_revenue)}",
                    ])
                    if not expenses.empty:
                        lines.extend([
                            f"Budgeted net profit ({source_file}): {self._format_sar(budget_revenue - budget_expenses, signed=True)}",
                            f"Actual net profit ({source_file}): {self._format_sar(actual_revenue - actual_expenses, signed=True)}",
                        ])

            if "amount_sar" in df.columns:
                amount_total = float(df["amount_sar"].sum())
                source_lower = source_file.lower()
                if "bank" in source_lower or "statement" in source_lower:
                    lines.append(f"Bank statement total ({source_file}): {self._format_sar(amount_total)}")
                elif "invoice" in query_lower or "revenue" in query_lower or "صفقات" in query_lower or "ايراد" in query_lower or "إيراد" in query:
                    if "record_type" in df.columns:
                        invoices = df[df["record_type"].astype(str).str.lower() == "invoice"]
                        pipeline = df[df["record_type"].astype(str).str.lower() == "pipeline"]
                        if not invoices.empty:
                            lines.append(
                                f"Invoice amount total before duplicate review ({source_file}): "
                                f"{self._format_sar(float(invoices['amount_sar'].sum()))}"
                            )
                        if not pipeline.empty:
                            stage_totals = pipeline.groupby("stage", dropna=True)["amount_sar"].sum()
                            for stage, value in stage_totals.items():
                                lines.append(f"Pipeline stage total - {stage} ({source_file}): {self._format_sar(float(value))}")
                    else:
                        lines.append(f"Amount total ({source_file}): {self._format_sar(amount_total)}")

        return lines

    def _build_anomalies(self, query: str) -> List[str]:
        query_lower = FinancialDataExtractor._normalize_arabic(query.lower())
        revenue_query = any(
            keyword in query_lower
            for keyword in ["invoice", "billing", "revenue", "collection", "فاتورة", "فواتير", "ايراد", "إيراد", "تحصيل", "صفقات"]
        )
        if not revenue_query:
            return []

        lines: List[str] = []
        for source_file, frame in self.frames:
            df = self._normalize_columns(frame)
            if not {"customer", "amount_sar"}.issubset(df.columns):
                continue

            invoice_df = df.copy()
            if "record_type" in invoice_df.columns:
                invoice_df = invoice_df[invoice_df["record_type"].astype(str).str.lower() == "invoice"]
            if invoice_df.empty:
                continue

            suspects = invoice_df[invoice_df.duplicated(subset=["customer", "amount_sar"], keep=False)]
            if suspects.empty:
                continue

            for (_, amount), group in suspects.groupby(["customer", "amount_sar"], dropna=False):
                ids = ", ".join(str(v) for v in group.get("id", pd.Series(dtype=str)).dropna().tolist())
                lines.append(
                    f"Potential duplicate invoice in {source_file}: {ids} have the same customer "
                    f"({group.iloc[0]['customer']}) and amount ({self._format_sar(float(amount))}); "
                    "verify before counting both as separate revenue."
                )

        return lines


# ============================================================================
# 4. OPTIMIZED Agentic RAG - Fast & Efficient
# ============================================================================

class FinancialRAGAgent:

    VIZ_KEYWORDS = [
        "chart", "visualiz", "plot", "graph", "draw", "pie", "bar", "line", "trend",
        "رسم بياني", "مخطط", "تمثيل بياني", "تصوير بياني", "تصور بياني",
        "رسم دائري", "رسم خطي", "رسم بالأعمدة", "رسم اعمدة", "رسم مبعثر",
        "ارسم", "إرسم", "اعرض بيانياً", "بيانياً", "بياني", "تصور",
    ]
    TIME_LABEL_MARKERS = (
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept",
        "oct", "nov", "dec", "quarter", "qtr", "month", "year", "fy",
        "يناير", "فبراير", "مارس", "ابريل", "أبريل", "مايو", "يونيو", "يوليو",
        "اغسطس", "أغسطس", "سبتمبر", "اكتوبر", "أكتوبر", "نوفمبر", "ديسمبر",
        "الربع", "ربع", "شهر", "سنة", "سنوي",
    )
    PIE_INTENT_MARKERS = (
        "share", "proportion", "breakdown", "distribution", "percentage",
        "percent", "of total", "split", "composition", "mix",
        "حصة", "نسبة", "نسب", "توزيع", "تفصيل", "تقسيم", "مكونات",
    )

    # Phrases that indicate the answer itself is reporting that no matching
    # data was found. If the text answer says this, we should not show a
    # chart that was built from a different, unrelated topic — that
    # contradicts the answer and misleads the user.
    NO_DATA_PHRASES = [
        "لا توجد بيانات", "لا يوجد بيانات", "لم يتم العثور", "لا يمكن رسم",
        "غير متوفرة في السياق", "غير متوفر في السياق", "لا تحتوي المستندات",
        "no data", "no relevant data", "not available in the context",
        "not found in the", "cannot be found in the", "unable to find",
        "does not contain", "no information", "insufficient data",
    ]

    FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")
    MAIN_MODEL = os.getenv("GROQ_MAIN_MODEL", "openai/gpt-oss-20b")

    def __init__(self, vector_db, chunks: Optional[List[Any]] = None):
        self.vector_db = vector_db
        self.chunks = chunks
        self._hybrid_retriever = HybridRetriever(vector_db, chunks) if chunks else None


        api_key = os.getenv("GROQ_API_KEY")
        secret_key = SecretStr(api_key) if api_key else None


        self.llm = ChatGroq(
            model=self.MAIN_MODEL,
            api_key=secret_key,
            temperature=0,
            max_retries=1,
            request_timeout=60,
        )
        self.fast_llm = ChatGroq(
            model=self.FAST_MODEL,
            api_key=secret_key,
            temperature=0,
            max_retries=1,
            request_timeout=30,
        )

  
        self._main_model_limiter = TPMRateLimiter(
            tpm_limit=int(os.getenv("GROQ_MAIN_MODEL_TPM_LIMIT", "8000")),
            safety_margin=0.9,
        )

        self._fast_model_limiter = TPMRateLimiter(
            tpm_limit=int(os.getenv("GROQ_FAST_MODEL_TPM_LIMIT", "30000")),
            safety_margin=0.9,
        )

    def process_query(
        self,
        query: str,
        chat_history: list = None,
        use_self_rag: bool = True,
        max_retries: int = 1,
    ) -> dict:
        _t0 = time.time()

        def _lap(label):
            nonlocal _t0
            now = time.time()
            print(f"⏱️ {label}: {now - _t0:.2f}s")
            _t0 = now

        if chat_history is None:
            chat_history = []
        max_retries = max(0, int(max_retries or 0))

        query_expander = QueryExpander(self.fast_llm) if self._hybrid_retriever else None

        retriever = CRAGRetriever(
            self.vector_db,
            self.fast_llm,
            hybrid_retriever=self._hybrid_retriever,
            query_expander=query_expander,
        )
        coordinator = WorkflowCoordinator(
            self.fast_llm,
            retriever,
            rate_limiter=self._fast_model_limiter,
        )

        evaluation = coordinator.evaluate(query, chat_history)
        _lap(f"Coordinator evaluation (needs_retrieval={evaluation.needs_retrieval})")

        if not evaluation.needs_retrieval:
            direct_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant for a financial document Q&A "
                           "system. Respond briefly and naturally in the same language "
                           "as the query. Do not invent any financial figures — if the "
                           "query actually needs document data, say you'd need to check "
                           "the documents."),
                ("human", "Chat History: {chat_history}\n\nQuery: {query}")
            ])
            formatted_direct_prompt = direct_prompt.format_messages(
                query=query,
                chat_history=chat_history,
            )
            direct_prompt_text = " ".join(str(m.content) for m in formatted_direct_prompt)
            self._fast_model_limiter.wait_if_needed(
                TPMRateLimiter.estimate_tokens(direct_prompt_text),
                label="Direct Answer",
            )
            response = self.fast_llm.invoke(formatted_direct_prompt)
            return {
                "answer": response.content.strip(),
                "source_pages": [],
                "confidence": "N/A",
                "verification": None,
                "relevant_docs_count": 0,
                "chart": None
            }

        # ------------------------------------------------------------------
        # PERF (latency only — zero change in logic/prompts/models):
        # aggregation_intent و chart_intent يعتمدون فقط على نص السؤال، مو
        # على المستندات المسترجعة من route(). سابقاً كانوا يشتغلون بالتسلسل
        # بعد ما route() يخلص (يعني نداءين إضافيين ينتظرون على طول نداء
        # الاسترجاع/CRAG الطويل). الحين الثلاثة (route + aggregation intent
        # + chart intent) يشتغلون بالتوازي على threads منفصلة، فالوقت الكلي
        # يصير أقرب لأطول عملية من الثلاث بدل مجموعهم. نفس البرومبتات،
        # نفس الموديلات (fast_llm)، نفس rate limiter، ونفس نتيجة الحساب
        # بالضبط — التغيير الوحيد هو وقت التنفيذ (wall-clock)، مو منطق
        # القرار نفسه.
        # ------------------------------------------------------------------
        aggregation_detector = AggregationIntentDetector(
            self.fast_llm,
            rate_limiter=self._fast_model_limiter,
        )
        chart_detector = ChartIntentDetector(
            self.fast_llm,
            rate_limiter=self._fast_model_limiter,
            fallback_keywords=self.VIZ_KEYWORDS,
        )

        with ThreadPoolExecutor(max_workers=3) as pre_executor:
            route_future = pre_executor.submit(coordinator.route, query)
            aggregation_future = pre_executor.submit(aggregation_detector.detect, query)
            chart_future = pre_executor.submit(chart_detector.detect, query)

            routed = route_future.result()
            aggregation_intent = aggregation_future.result()
            needs_chart = chart_future.result()

        _lap("Retrieval TOTAL (expansion + hybrid search + CRAG grading) + intent detection (parallel)")
        print(f"📊 Chart intent detected: {needs_chart}")

        if routed["instruction"].action == "REPORT_NOT_FOUND":
            return {
                "answer": "No relevant documents found for your query.",
                "source_pages": [],
                "confidence": "Low",
                "verification": None,
                "relevant_docs_count": 0,
                "chart": None
            }

        relevant_docs = routed["documents"]

        def _source_file(doc) -> str:
            metadata = doc.metadata or {}
            return str(metadata.get("source_file") or metadata.get("source") or "unknown")

        def _context_label(doc) -> str:
            metadata = doc.metadata or {}
            source_file = _source_file(doc)
            if metadata.get("table_full_context"):
                return f"[Source: {source_file} | Full table]"
            if metadata.get("page") is not None:
                return f"[Source: {source_file} | Page {metadata.get('page')}]"
            if metadata.get("sheet_name"):
                return f"[Source: {source_file} | Sheet: {metadata.get('sheet_name')}]"
            row = metadata.get("row")
            if row is not None:
                return f"[Source: {source_file} | Row {row}]"
            return f"[Source: {source_file}]"

        context = "\n\n".join([
            f"{_context_label(doc)}\n{doc.page_content}"
            for doc in relevant_docs
        ])

        deterministic_block = DeterministicContextAnalyzer(
            relevant_docs
        ).build_verified_context_block(query, aggregation_intent)
        if deterministic_block:
            context = f"{deterministic_block}\n\n{context}"
            print(f"🧮 Injected deterministic context block:\n{deterministic_block}")

      
        print(f"📄 Context المرسل للموديل ({len(context)} حرف من {len(relevant_docs)} مستند):")
        print(f"   {context[:300]}{'...' if len(context) > 300 else ''}")

        def _source_label(doc) -> Optional[str]:
            metadata = doc.metadata or {}
            source_file = _source_file(doc)
            if metadata.get("table_full_context"):
                return source_file
            page = metadata.get('page')
            if page is not None:
                return f"{source_file} p.{page}"
            sheet = metadata.get('sheet_name')
            if sheet:
                return f"{source_file} / Sheet: {sheet}"
            return source_file

        source_pages = sorted({
            label
            for label in (_source_label(doc) for doc in relevant_docs)
            if label is not None
        })

        labeled_source_texts = [
            f"{_context_label(doc)}\n{doc.page_content}"
            for doc in relevant_docs
        ]
        if deterministic_block:
            labeled_source_texts.insert(0, deterministic_block)

        # ========================================================================
        # ========================================================================
        refine_engine = SelfRefiningAnswerEngine(
            self.llm,
            verifier_llm=self.fast_llm,
            max_refinement_attempts=max_retries, pass_threshold=7,
            rate_limiter=self._main_model_limiter,          # للموديل الرئيسي فقط (Generation + Refinement)
            verifier_rate_limiter=self._fast_model_limiter,  # منفصل تماماً لـ fast_llm
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            answer_future = executor.submit(
                refine_engine.run,
                query=query,
                context=context,
                chat_history=chat_history,
                source_texts=labeled_source_texts,
                chart_requested=needs_chart,
                use_self_rag=use_self_rag,
            )

            chart_future = None
            if needs_chart:
                # Reuse the same CRAG-graded documents as the text answer so
                # the chart can never "find" data on a topic the answer
                # itself said was missing.
                chart_future = executor.submit(self._build_chart, query, relevant_docs)

            refine_result = answer_future.result()
            chart_data = chart_future.result() if chart_future is not None else None
        _lap(f"Answer generation + verification (attempts={refine_result['attempts_made']})" + (" + chart (parallel)" if needs_chart else ""))

        answer = refine_result["answer"]
        verification = refine_result["verification"]

        # If the text answer itself states that no matching data was found,
        # never show a chart — it would necessarily be built from an
        # unrelated topic and would contradict/mislead relative to the text.
        if chart_data and self._answer_indicates_no_data(answer):
            print("⚠️ الإجابة تفيد بعدم وجود بيانات مطابقة — سيتم إخفاء الرسم البياني لتفادي التناقض")
            chart_data = None

        if verification is None:
            confidence = "N/A"
        else:
            confidence = "High" if verification.get("rating", 0) >= 8 else "Medium" if verification.get("rating", 0) >= 5 else "Low"

        return {
            "answer": answer,
            "source_pages": source_pages,
            "confidence": confidence,
            "verification": verification,
            "relevant_docs_count": len(relevant_docs),
            "source_texts": labeled_source_texts,
            "chart": chart_data,
            "self_refine_attempts": refine_result["attempts_made"],
            "self_refine_converged": refine_result["self_refine_converged"],
        }

    @classmethod
    def _answer_indicates_no_data(cls, answer: str) -> bool:
        answer_lower = (answer or "").lower()
        return any(phrase in answer_lower for phrase in cls.NO_DATA_PHRASES)

    def _build_chart(self, query: str, relevant_docs: Optional[List[Any]] = None) -> Optional[Dict[str, Any]]:
        try:
            _chart_t0 = time.time()
            extractor = FinancialDataExtractor(self.vector_db, self.fast_llm)
            df = extractor.extract_data_from_query(query, docs=relevant_docs)
            print(f"⏱️ Chart data extraction: {time.time() - _chart_t0:.2f}s")
            print(f"📊 DataFrame for visualization:\n{df.head()}")

            if df.empty:
                return None

            try:
                fast_llm = getattr(self, "fast_llm", None)
                if fast_llm is None or not hasattr(fast_llm, "with_structured_output"):
                    raise RuntimeError("fast_llm unavailable for chart type selection")

                selector = ChartTypeSelector(
                    fast_llm,
                    rate_limiter=getattr(self, "_fast_model_limiter", None),
                )
                decision = selector.select(query, df)
                chart_type = decision.chart_type
                print(f"📊 Chart type chosen: {chart_type} ({decision.reason})")
            except Exception as e:
                chart_type = self._fallback_chart_type(query, df)
                print(f"⚠️ Chart type selector unavailable: {e} — fallback chose {chart_type}")

            validated_chart_type = self._validate_chart_type(chart_type, query, df)
            if validated_chart_type != chart_type:
                print(
                    f"📊 Chart type adjusted: {chart_type} -> {validated_chart_type} "
                    "(data shape guard)"
                )
                chart_type = validated_chart_type

            if chart_type == "bar":
                fig = ChartGenerator.create_bar_chart(df, x="label", y="value", title=query)
            elif chart_type == "line":
                fig = ChartGenerator.create_line_chart(df, x="label", y="value", title=query)
            elif chart_type == "pie":
                fig = ChartGenerator.create_pie_chart(df, names="label", values="value", title=query)
            else:
                fig = ChartGenerator.create_area_chart(df, x="label", y="value", title=query)

            if "currency" in df.columns and set(df["currency"].dropna()) == {"%"}:
                fig.update_yaxes(title_text="النسبة (%)")
                fig.update_traces(
                    texttemplate="%{y:g}%",
                    textposition="auto",
                    hovertemplate="%{x}<br>%{y:g}%<extra></extra>",
                )

            return {
                "success": True,
                "chart": fig.to_json(),
                "title": query,
                "data_preview": df.to_dict(orient="records")
            }
        except Exception as e:
            print(f"⚠️ Chart generation error: {e}")
            return {"success": False, "error": str(e)}

    def _fallback_chart_type(self, query: str, df: pd.DataFrame) -> str:
        query_lower = query.lower()

        if FinancialDataExtractor._is_improvement_percentage_query(query):
            return "bar"

        # Honour an explicitly requested chart type before applying heuristics.
        if any(kw in query_lower for kw in ["area chart", "area graph", "مساحي", "مساحة"]):
            return "area"
        if any(kw in query_lower for kw in ["pie", "دائري", "دائرة"]):
            return "pie"
        if any(kw in query_lower for kw in ["line", "خطي"]):
            return "line"
        if any(kw in query_lower for kw in ["bar", "أعمدة", "اعمدة", "عمودي"]):
            return "bar"

        if any(kw in query_lower for kw in [
            "trend", "over time", "quarterly", "yearly", "monthly",
            "اتجاه", "عبر الزمن", "ربع سنوي", "ربعي", "سنوي", "شهري", "تطور",
        ]):
            return "line"
        if any(kw in query_lower for kw in [
            "compare", "comparison", "breakdown", "share", "distribution",
            "مقارنة", "توزيع", "تفصيل", "حصة",
        ]):
            return "pie" if len(df) <= 6 else "bar"

        return "bar"

    def _validate_chart_type(self, chart_type: str, query: str, df: pd.DataFrame) -> str:
        if chart_type not in {"bar", "line", "pie", "area"}:
            return self._fallback_chart_type(query, df)

        if FinancialDataExtractor._is_improvement_percentage_query(query):
            return "bar"

        if chart_type in {"line", "area"} and not self._labels_look_time_ordered(df):
            return "bar"

        if chart_type == "pie" and not self._pie_chart_is_reasonable(query, df):
            return "bar"

        return chart_type

    @classmethod
    def _labels_look_time_ordered(cls, df: pd.DataFrame) -> bool:
        if "label" not in df.columns:
            return False

        labels = [str(label).strip() for label in df["label"].dropna().tolist()]
        if len(labels) < 2:
            return False

        matches = sum(1 for label in labels if cls._is_time_label(label))
        required = max(2, int(np.ceil(len(labels) * 0.5)))
        return matches >= required

    @classmethod
    def _is_time_label(cls, label: str) -> bool:
        normalized = FinancialDataExtractor._normalize_arabic(label.lower())

        if re.search(r"\b(?:q[1-4]|qtr\s*[1-4]|quarter\s*[1-4]|fy\s*\d{2,4})\b", normalized):
            return True
        if re.search(r"\b(?:19|20)\d{2}\b", normalized):
            return True
        if re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", normalized):
            return True

        return any(marker in normalized for marker in cls.TIME_LABEL_MARKERS)

    def _pie_chart_is_reasonable(self, query: str, df: pd.DataFrame) -> bool:
        query_lower = FinancialDataExtractor._normalize_arabic(query.lower())
        if not any(marker in query_lower for marker in self.PIE_INTENT_MARKERS):
            return False
        if len(df) < 2 or len(df) > 6:
            return False
        if "value" not in df.columns:
            return False
        if self._labels_look_time_ordered(df):
            return False

        values = pd.to_numeric(df["value"], errors="coerce")
        if values.isna().any():
            return False
        return bool((values >= 0).all())

    @classmethod
    def _needs_chart(cls, query: str) -> bool:
        return _has_chart_keyword(query, cls.VIZ_KEYWORDS)

    def _clean_dataframe(self, data: list) -> pd.DataFrame:
        cleaned = []

        for item in data:
            if not isinstance(item, dict):
                continue

            label = item.get('label')
            if label is None or str(label).strip() == '':
                continue
            label = str(label).strip()

            value = item.get('value')
            if value is None or str(value).strip() == '':
                continue

            value_str = str(value).strip()
            value_str = value_str.replace(',', '')
            value_str = value_str.replace('﷼', '')
            value_str = value_str.replace('$', '')
            value_str = value_str.replace(' ', '')

            value_str = re.sub(r'(billion|million|trillion|bn|mn|tn|مليار|مليون)', '', value_str, flags=re.IGNORECASE)
            value_str = value_str.strip()

            if value_str == '':
                continue

            try:
                numeric_value = float(value_str)
                cleaned.append({'label': label, 'value': numeric_value})
            except (ValueError, TypeError):
                print(f"  ⚠️ Skipping invalid value: label='{label}', value='{value}'")
                continue

        if not cleaned:
            return pd.DataFrame()

        df = pd.DataFrame(cleaned)
        df = df.drop_duplicates(subset=['label'])

        return df
