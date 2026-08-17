import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi


# ============================================================================
# Hybrid Retriever — BM25 (لفظي) + Vector Search (دلالي) عبر RRF
# ============================================================================
#
# يدعم الاسترجاع بعدة صياغات للسؤال بنفس الوقت (متكامل مع QueryExpander)،
# ويدمج كل النتائج بـ Reciprocal Rank Fusion واحد موحّد قبل ما تدخل على
# CRAGRetriever للفلترة الدلالية النهائية.


def _tokenize(text: str) -> List[str]:
    """
    تقسيم بسيط يدعم عربي/إنجليزي:
    - يشيل التشكيل العربي (تنوين، فتحة، ضمة...الخ) عشان ما يأثر بالتطابق
    - يشيل علامات الترقيم
    - يفكك على المسافات
    ملاحظة: لو محتواك عربي بكثافة عالية، فكّر تستبدل هذا لاحقاً بمُجذّع
    (stemmer) عربي مخصص مثل ISRIStemmer أو camel-tools لتحسين إضافي.
    """
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)  # إزالة التشكيل
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    text = re.sub(r'\b(\d[\d,]*(?:\.\d+)?)\s*(million|mn|m)\b', _million_token, text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\d[\d,]*(?:\.\d+)?)\s*(billion|bn|b)\b', _billion_token, text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\d[\d,]*(?:\.\d+)?)\s*(thousand|k)\b', _thousand_token, text, flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    tokens = text.lower().split()
    expanded_tokens = list(tokens)
    for token in tokens:
        if re.fullmatch(r'\d[\d,]*(?:\.\d+)?', token):
            expanded_tokens.append(token.replace(",", ""))
    return expanded_tokens


def _scaled_number_token(match, factor: int) -> str:
    value = float(match.group(1).replace(",", "")) * factor
    return f" {int(value) if value.is_integer() else value:g} "


def _million_token(match) -> str:
    return _scaled_number_token(match, 1_000_000)


def _billion_token(match) -> str:
    return _scaled_number_token(match, 1_000_000_000)


def _thousand_token(match) -> str:
    return _scaled_number_token(match, 1_000)


class HybridRetriever:
    """
    يبني فهرس BM25 مرة وحدة فوق نفس الـ chunks اللي بُني منها الـ FAISS
    vector_db، وبعدين يدمج نتائج البحث اللفظي والدلالي عبر RRF.

    مهم: `documents` لازم تكون نفس قائمة الـ Document objects (بعد الـ
    chunking) اللي استُخدمت لبناء الـ vector_db بالضبط — عشان النتائج
    تتطابق. أسهل طريقة: خزّنها وقت `build_vector_db` (مثلاً كـ pickle
    بجانب الـ FAISS index) وحمّلها هنا.
    """

    def __init__(self, vector_db, documents: List[Any], rrf_k: int = 60):
        self.vector_db = vector_db
        self.documents = documents
        self.rrf_k = rrf_k

        tokenized_corpus = [_tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # مفتاح فريد لكل مستند (نستخدمه للدمج بدل الاعتماد على المحتوى كامل)
        self._doc_keys = [self._make_key(doc, i) for i, doc in enumerate(documents)]

    @staticmethod
    def _make_key(doc, fallback_idx: int) -> str:
        meta = doc.metadata or {}
        source = meta.get("source", "")
        page = meta.get("page", meta.get("row", fallback_idx))
        sheet = meta.get("sheet_name", "")
        return f"{source}|{sheet}|{page}|{fallback_idx}"

    def _bm25_ranked_indices(self, query: str, top_n: int) -> List[int]:
        scores = self.bm25.get_scores(_tokenize(query))
        if not len(scores):
            return []
        ranked = np.argsort(scores)[::-1][:top_n]
        return [int(i) for i in ranked if scores[i] > 0]

    def _vector_ranked_docs(self, query: str, top_n: int) -> List[Any]:
        try:
            return self.vector_db.similarity_search(query, k=top_n)
        except Exception as e:
            print(f"⚠️ Vector search error for query '{query[:50]}...': {e}")
            return []

    def _per_query_hits(self, query: str, k_max: int):
        """
        كل الشغل الخاص بصياغة سؤال واحدة (BM25 + vector) — يُنفّذ كوحدة
        واحدة على thread منفصل، عشان الصياغات المتعددة (أصلي + موسّعة من
        QueryExpander) تشتغل بالتوازي بدل التسلسل. يرجّع قائمة hits على
        شكل (key, doc, rank) لكل من BM25 والـ vector، بدون أي حساب RRF
        هنا — الدمج (اللي لازم يصير بالتسلسل على نتيجة كل الـ threads)
        يبقى بمكان واحد بالدالة اللي تستدعي هذي.
        """
        hits = []

        bm25_idx = self._bm25_ranked_indices(query, top_n=k_max)
        for rank, idx in enumerate(bm25_idx):
            hits.append((self._doc_keys[idx], self.documents[idx], rank))

        vector_docs = self._vector_ranked_docs(query, top_n=k_max)
        for rank, doc in enumerate(vector_docs):
            key = self._make_key(doc, fallback_idx=hash(doc.page_content[:200]))
            hits.append((key, doc, rank))

        return hits

    def _fuse(self, queries: List[str], k_max: int):
        """
        PERF FIX: قبل كذا، كل query بالقائمة كان يُعالج بالتسلسل (BM25 كامل
        على الكوربص + نداء FAISS منفصل لكل واحدة). مع 3-4 صياغات من
        QueryExpander، هذا يعني تكرار العملية كاملة 3-4 مرات بالتتابع قبل
        حتى ما توصل لأي نداء LLM. الحين تشتغل كل الصياغات بالتوازي عبر
        ThreadPoolExecutor، ثم يصير دمج RRF بالتسلسل على النتائج الجاهزة —
        نفس الحساب الرياضي بالضبط (لا تغيير على النتيجة النهائية أو
        الترتيب)، بس الوقت الكلي يصير أقرب لأطول query لا مجموعهم كلهم.
        """
        rrf_scores: Dict[str, float] = {}
        doc_lookup: Dict[str, Any] = {}

        max_workers = min(len(queries), 8) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            all_hits = list(executor.map(lambda q: self._per_query_hits(q, k_max), queries))

        for hits in all_hits:
            for key, doc, rank in hits:
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                doc_lookup[key] = doc

        fused = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
        return fused, doc_lookup

    def retrieve_single(self, query: str, k_max: int) -> List[Any]:
        """استرجاع hybrid لسؤال واحد فقط (بدون توسعة)."""
        return self.retrieve_multi([query], k_max)

    def retrieve_multi(self, queries: List[str], k_max: int) -> List[Any]:
        """
        استرجاع hybrid لعدة صياغات للسؤال (أصلي + موسّع)، مدمجة كلها بـ
        RRF واحد موحّد. كل صياغة تساهم بترتيبها اللفظي والدلالي، والمستند
        اللي يظهر بأكثر من صياغة/طريقة يرتفع ترتيبه تلقائياً.
        """
        fused, doc_lookup = self._fuse(queries, k_max)
        top_docs = [doc_lookup[key] for key, _ in fused[:k_max]]

        print(f"🔀 Hybrid RRF: دمج {len(queries)} صياغة سؤال -> {len(top_docs)} مستند مرشّح")
        return top_docs

    def retrieve_with_scores(self, queries: List[str], k_max: int):
        """
        نفس retrieve_multi لكن يرجّع أيضاً درجات RRF (بدل درجات cosine
        الأصلية) عشان يستمر AdaptiveRetrievalDepth.detect_elbow يشتغل
        بنفس المنطق الموجود بـ CRAGRetriever.
        """
        fused, doc_lookup = self._fuse(queries, k_max)
        top = fused[:k_max]
        docs = [doc_lookup[key] for key, _ in top]
        scores = [score for _, score in top]
        return docs, scores
