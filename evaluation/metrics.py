"""Financial retrieval evaluation metrics.

This module provides pure helper functions for scoring financial-document
retrieval and extraction results using precision, recall, F1-score, and
retrieval accuracy.
"""

from __future__ import annotations

import json
import re
import sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

DEFAULT_HISTORY_DB_PATH = Path(__file__).resolve().parents[1] / "src" / "chat_history.db"
RETRIEVED_FIELD_ALIASES = (
    "retrieved",
    "retrieved_items",
    "retrieved_financial_items",
    "extracted",
    "extracted_items",
    "extracted_fields",
    "extracted_figures",
    "generated_answer",
    "answer",
)
RELEVANT_FIELD_ALIASES = (
    "relevant",
    "relevant_items",
    "relevant_financial_items",
    "expected",
    "expected_items",
    "expected_financial_items",
    "ground_truth",
    "ground_truth_items",
    "reference",
    "reference_answer",
    "reference_items",
    "source_texts",
    "source_chunks",
    "retrieved_context",
)
EXPLICIT_RELEVANT_FIELD_ALIASES = (
    "relevant",
    "relevant_items",
    "relevant_financial_items",
    "expected",
    "expected_items",
    "expected_financial_items",
    "ground_truth",
    "ground_truth_items",
    "reference",
    "reference_answer",
    "reference_items",
)
SOURCE_SUPPORT_FIELD_ALIASES = (
    "source_texts",
    "source_chunks",
    "retrieved_context",
)
SAMPLE_RETRIEVAL_RESULTS = [
    (
        {
            "revenue: 100m SAR",
            "net income: 15m SAR",
            "cash flow: SAR 22 million",
        },
        {
            "revenue: 100m SAR",
            "net income: 15m SAR",
            "cash flow: 22,000,000 SAR",
        },
    )
]
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_CURRENCY_REPLACEMENTS = {
    "$": " usd ",
    "€": " eur ",
    "£": " gbp ",
    "ريال": " sar ",
    "ر.س": " sar ",
    "٪": "%",
}
_NUMBER_RE = re.compile(
    r"\(?[-+]?\d[\d,]*(?:\.\d+)?\)?(?:\s*(?:%|percent|percentage|million|mn|m|billion|bn|b|thousand|k))?",
    re.IGNORECASE,
)
_UNIT_FACTORS = {
    "k": Decimal("1000"),
    "thousand": Decimal("1000"),
    "m": Decimal("1000000"),
    "mn": Decimal("1000000"),
    "million": Decimal("1000000"),
    "b": Decimal("1000000000"),
    "bn": Decimal("1000000000"),
    "billion": Decimal("1000000000"),
}
_NO_DATA_RESPONSE_RE = re.compile(
    r"(?:"
    r"does not include any financial information|"
    r"no financial information|"
    r"no relevant documents|"
    r"unable to generate a response|"
    r"not provided in the context|"
    r"لا يتضمن السياق|"
    r"غير متوفر|"
    r"لا توجد بيانات"
    r")",
    re.IGNORECASE,
)
_SCALE_DENOMINATOR_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(?:/|\bمن\b|\bout\s+of\b)\s*[45]\b",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "with",
    "very",
    "high",
    "low",
    "medium",
    "هو",
    "هي",
    "من",
    "في",
    "على",
    "عن",
    "إلى",
    "الى",
    "مع",
    "هذا",
    "هذه",
    "ذلك",
    "تلك",
    "هل",
    "كيف",
    "ما",
    "أي",
    "اي",
    "أو",
    "او",
    "و",
    "جدا",
    "جداً",
    "مرتفع",
    "مرتفعة",
    "ممتاز",
    "جيد",
    "البرنامج",
    "المشاركين",
}


def precision(true_positives: int, false_positives: int) -> float:
    """Precision (الدقة): measure how correct and reliable retrieved financial information is.

    It is the proportion of retrieved items that are actually relevant or
    correct, free of irrelevant results.

    Formula: Precision = True Positives / (True Positives + False Positives).
    Returns 0.0 when the denominator is 0.
    """
    denominator = true_positives + false_positives
    if denominator == 0:
        return 0.0
    return true_positives / denominator


def recall(true_positives: int, false_negatives: int) -> float:
    """Recall (الاستدعاء): measure the system's ability to cover required financial data.

    It captures how well the system finds all required financial data present
    in the full set of source documents.

    Formula: Recall = True Positives / (True Positives + False Negatives).
    Returns 0.0 when the denominator is 0.
    """
    denominator = true_positives + false_negatives
    if denominator == 0:
        return 0.0
    return true_positives / denominator


def f1_score(precision_value: float, recall_value: float) -> float:
    """F1-Score (مؤشر التوازن): harmonic mean balancing Precision and Recall.

    It combines Precision (الدقة), reflecting accuracy and reliability, with
    Recall (الاستدعاء), reflecting coverage of required financial data, into a
    single number that shows how stable the model is between accuracy and
    coverage.

    Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall).
    Returns 0.0 when the denominator is 0.
    """
    denominator = precision_value + recall_value
    if denominator == 0:
        return 0.0
    return 2 * (precision_value * recall_value) / denominator


def accuracy(true_positives: int, false_positives: int, false_negatives: int) -> float:
    """Accuracy (الصحة): measure overall correctness for retrieved financial items.

    In a retrieval/extraction setting there are no explicit true negatives, so
    this uses the available comparison universe: correctly retrieved financial
    items divided by all retrieved-or-relevant items.

    Formula: Accuracy = True Positives / (True Positives + False Positives + False Negatives).
    Returns 0.0 when the denominator is 0.
    """
    denominator = true_positives + false_positives + false_negatives
    if denominator == 0:
        return 0.0
    return true_positives / denominator


def analyze_retrieval_errors(retrieved: set[str], relevant: set[str]) -> dict[str, Any]:
    """Analyze false positives/false negatives for financial retrieval errors.

    Precision (الدقة), Recall (الاستدعاء), F1-Score (مؤشر التوازن), and
    Accuracy (الصحة) depend on correctly matching equivalent financial items.
    This helper reports exact-match misses and categorizes likely causes such
    as matching/normalization, retrieval/extraction, or ground-truth labeling.
    """
    exact_false_positives = sorted(retrieved - relevant)
    exact_false_negatives = sorted(relevant - retrieved)

    normalized_retrieved = {_normalize_financial_item(item): item for item in retrieved}
    normalized_relevant = {_normalize_financial_item(item): item for item in relevant}

    normalized_false_positives = set(normalized_retrieved) - set(normalized_relevant)
    normalized_false_negatives = set(normalized_relevant) - set(normalized_retrieved)

    categories: list[dict[str, str]] = []
    matched_normalized_keys = set(normalized_retrieved) & set(normalized_relevant)

    for key in sorted(matched_normalized_keys):
        retrieved_item = normalized_retrieved[key]
        relevant_item = normalized_relevant[key]
        if retrieved_item != relevant_item:
            categories.append(
                {
                    "category": "matching/normalization",
                    "retrieved_item": retrieved_item,
                    "relevant_item": relevant_item,
                    "reason": "Same financial label/value after currency, comma, and unit normalization.",
                }
            )

    for key in sorted(normalized_false_positives):
        categories.append(
            {
                "category": "retrieval/extraction false positive",
                "retrieved_item": normalized_retrieved[key],
                "relevant_item": "",
                "reason": "Retrieved item has no matching relevant item after normalization.",
            }
        )

    for key in sorted(normalized_false_negatives):
        categories.append(
            {
                "category": "retrieval/chunking false negative",
                "retrieved_item": "",
                "relevant_item": normalized_relevant[key],
                "reason": "Relevant item was not retrieved after normalization.",
            }
        )

    if not exact_false_positives and not exact_false_negatives:
        likely_cause = "no_error"
    elif not normalized_false_positives and not normalized_false_negatives:
        likely_cause = "matching/normalization"
    else:
        likely_cause = "retrieval_or_extraction"

    return {
        "retrieved": sorted(retrieved),
        "relevant": sorted(relevant),
        "exact_false_positives": exact_false_positives,
        "exact_false_negatives": exact_false_negatives,
        "likely_cause": likely_cause,
        "categorized_errors": categories,
    }


def evaluate(retrieved: set[str], relevant: set[str], *, normalize: bool = True) -> dict[str, float | int]:
    """Evaluate retrieved financial items against relevant ground truth items.

    Precision (الدقة): proportion of retrieved financial items that are
    relevant or correct. Recall (الاستدعاء): ability to capture all required
    financial data in the source documents. F1-Score (مؤشر التوازن): harmonic
    mean that balances accuracy and coverage. Accuracy (الصحة): overall
    correctness across retrieved-or-relevant financial items.

    ``retrieved`` contains items the system extracted or returned.
    ``relevant`` contains ground-truth items that should have been found.
    Set ``normalize=False`` to reproduce exact string-match baselines.
    """
    if normalize:
        retrieved_items = {_normalize_financial_item(item) for item in retrieved}
        relevant_items = {_normalize_financial_item(item) for item in relevant}
    else:
        retrieved_items = set(retrieved)
        relevant_items = set(relevant)

    true_positives = len(retrieved_items & relevant_items)
    false_positives = len(retrieved_items - relevant_items)
    false_negatives = len(relevant_items - retrieved_items)

    precision_value = precision(true_positives, false_positives)
    recall_value = recall(true_positives, false_negatives)
    f1_value = f1_score(precision_value, recall_value)
    accuracy_value = accuracy(true_positives, false_positives, false_negatives)

    return {
        "precision": precision_value,
        "recall": recall_value,
        "f1_score": f1_value,
        "accuracy": accuracy_value,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def evaluate_batch(results: list[tuple[set[str], set[str]]], *, normalize: bool = True) -> dict[str, Any]:
    """Evaluate multiple financial documents with micro-averaged metrics.

    Precision (الدقة): proportion of retrieved financial items that are
    relevant or correct across all documents. Recall (الاستدعاء): ability to
    capture all required financial data across all source documents. F1-Score
    (مؤشر التوازن): harmonic mean that balances accuracy and coverage.
    Accuracy (الصحة): overall correctness across retrieved-or-relevant
    financial items.

    ``results`` is a list of ``(retrieved, relevant)`` pairs, one per document.
    The return value includes micro-averaged metrics and a per-document
    breakdown. Set ``normalize=False`` to reproduce exact string-match
    baselines.
    """
    per_document = [evaluate(retrieved, relevant, normalize=normalize) for retrieved, relevant in results]

    true_positives = sum(int(document["true_positives"]) for document in per_document)
    false_positives = sum(int(document["false_positives"]) for document in per_document)
    false_negatives = sum(int(document["false_negatives"]) for document in per_document)

    precision_value = precision(true_positives, false_positives)
    recall_value = recall(true_positives, false_negatives)
    f1_value = f1_score(precision_value, recall_value)
    accuracy_value = accuracy(true_positives, false_positives, false_negatives)

    return {
        "precision": precision_value,
        "recall": recall_value,
        "f1_score": f1_value,
        "accuracy": accuracy_value,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "per_document": per_document,
    }


def print_overall_evaluation(evaluations: dict[str, Any]) -> None:
    """Print a compact overall evaluation matrix from an evaluation output dict."""
    row = evaluations.get("current_overall_evaluation")
    if not (
        isinstance(row, dict)
        and {"precision", "recall", "f1_score", "accuracy"}.issubset(row)
    ):
        return

    print("\nOverall Evaluation")
    print("-" * 106)
    print(
        f"{'Run':<34} {'Precision':>10} {'Recall':>10} {'F1':>10} "
        f"{'Accuracy':>10} {'TP':>6} {'FP':>6} {'FN':>6}"
    )
    print("-" * 106)
    print(
        f"{'Current overall':<34} "
        f"{float(row['precision']):>10.3f} "
        f"{float(row['recall']):>10.3f} "
        f"{float(row['f1_score']):>10.3f} "
        f"{float(row['accuracy']):>10.3f} "
        f"{int(row.get('true_positives', 0)):>6} "
        f"{int(row.get('false_positives', 0)):>6} "
        f"{int(row.get('false_negatives', 0)):>6}"
    )
    print("-" * 106)


def load_history_results(
    db_path: str | Path = DEFAULT_HISTORY_DB_PATH,
    *,
    session_id: str | None = None,
    retrieved_key: str | None = None,
    relevant_key: str | None = None,
    apply_precision_filters: bool = True,
) -> list[tuple[set[str], set[str]]]:
    """Load retrieved/relevant pairs from the chat history database.

    Precision (الدقة), Recall (الاستدعاء), F1-Score (مؤشر التوازن), and
    Accuracy (الصحة) require both retrieved financial items and relevant
    ground-truth items. This function reads ``chat_entries.entry_json`` from
    the SQLite history DB and returns only entries that contain both sides.

    By default it looks for common field names such as ``retrieved``,
    ``extracted_items``, ``answer``, ``relevant``, ``expected``, and
    ``ground_truth``, and retrieved source fields such as ``source_texts`` or
    ``source_chunks``. Pass ``retrieved_key`` and ``relevant_key`` to force
    specific JSON field names.

    When only source chunks are available as the reference side, the default
    ``apply_precision_filters=True`` applies an item-level rerank/grounding
    pass: it keeps only answer items whose material numbers are directly
    traceable to a single source row/chunk and whose text matches the query
    intent. Use ``apply_precision_filters=False`` to reproduce the legacy
    baseline that passed all extracted answer items into scoring.
    """
    records = load_history_evaluation_records(
        db_path,
        session_id=session_id,
        retrieved_key=retrieved_key,
        relevant_key=relevant_key,
        apply_precision_filters=apply_precision_filters,
    )
    return [(record["retrieved"], record["relevant"]) for record in records]


def load_jsonl_results(
    path: str | Path,
    *,
    retrieved_key: str | None = None,
    relevant_key: str | None = None,
) -> list[tuple[set[str], set[str]]]:
    """Load retrieved/reference item pairs from a JSONL fixture."""
    resolved_path = Path(path)
    if not resolved_path.exists():
        return []

    pairs: list[tuple[set[str], set[str]]] = []
    with resolved_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue

            retrieved_value = _field_value(entry, retrieved_key, RETRIEVED_FIELD_ALIASES)
            relevant_value = _field_value(entry, relevant_key, RELEVANT_FIELD_ALIASES)
            if retrieved_value is _MISSING or relevant_value is _MISSING:
                continue
            pairs.append((_to_item_set(retrieved_value), _to_item_set(relevant_value)))
    return pairs


def load_history_evaluation_records(
    db_path: str | Path = DEFAULT_HISTORY_DB_PATH,
    *,
    session_id: str | None = None,
    retrieved_key: str | None = None,
    relevant_key: str | None = None,
    apply_precision_filters: bool = True,
) -> list[dict[str, Any]]:
    """Load evaluable history rows with metadata used for error analysis."""
    resolved_path = Path(db_path)
    if not resolved_path.exists():
        return []

    with sqlite3.connect(resolved_path) as connection:
        connection.row_factory = sqlite3.Row
        if not _has_chat_entries_table(connection):
            return []
        query = "SELECT id, entry_json FROM chat_entries"
        params: tuple[str, ...] = ()
        if session_id is not None:
            query += " WHERE session_id = ?"
            params = (session_id,)
        query += " ORDER BY id ASC"
        rows = connection.execute(query, params).fetchall()

    records: list[dict[str, Any]] = []
    for row in rows:
        try:
            entry = json.loads(str(row["entry_json"]))
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(entry, dict):
            continue

        retrieved_value = _field_value(entry, retrieved_key, RETRIEVED_FIELD_ALIASES)
        if retrieved_value is _MISSING:
            continue

        retrieved_items = _to_item_set(retrieved_value)
        original_retrieved_items = set(retrieved_items)
        source_value = _field_value(entry, None, SOURCE_SUPPORT_FIELD_ALIASES)
        relevant_value = _field_value(entry, relevant_key, EXPLICIT_RELEVANT_FIELD_ALIASES)
        relevant_is_source_context = (
            relevant_key is None
            and source_value is not _MISSING
            and relevant_value is not _MISSING
            and relevant_value == source_value
        )
        if relevant_value is not _MISSING and not relevant_is_source_context:
            records.append(
                {
                    "entry_id": int(row["id"]),
                    "question": str(entry.get("question", "")),
                    "retrieved": _dedupe_financial_items(retrieved_items) if apply_precision_filters else retrieved_items,
                    "relevant": _to_item_set(relevant_value),
                    "original_retrieved": original_retrieved_items,
                    "source_items": set(),
                    "source_texts": [],
                    "precision_filter_applied": False,
                }
            )
            continue

        if source_value is _MISSING:
            continue

        source_items = _to_item_set(source_value)
        question = str(entry.get("question", ""))
        source_relevant_items = {
            item
            for item in retrieved_items
            if _is_supported_by_source_items(item, source_items)
        }
        if apply_precision_filters:
            decisions = [
                _grounding_decision(item, source_items, question)
                for item in retrieved_items
            ]
            supported_items = {
                str(decision["retrieved_item"])
                for decision in decisions
                if decision["accepted"]
            }
            retrieved_items = _dedupe_financial_items(supported_items)
            relevant_items = _dedupe_financial_items(source_relevant_items)
        else:
            relevant_items = source_relevant_items

        records.append(
            {
                "entry_id": int(row["id"]),
                "question": question,
                "retrieved": retrieved_items,
                "relevant": relevant_items,
                "original_retrieved": original_retrieved_items,
                "source_items": source_items,
                "source_texts": list(_to_raw_text_list(source_value)),
                "precision_filter_applied": apply_precision_filters,
            }
        )

    return records


def evaluate_history_db(
    db_path: str | Path = DEFAULT_HISTORY_DB_PATH,
    *,
    session_id: str | None = None,
    retrieved_key: str | None = None,
    relevant_key: str | None = None,
    normalize: bool = True,
    apply_precision_filters: bool = True,
) -> dict[str, Any]:
    """Evaluate financial retrieval metrics from the chat history database.

    Precision (الدقة): correctness of retrieved history items. Recall
    (الاستدعاء): coverage of expected ground-truth items in history.
    F1-Score (مؤشر التوازن): balance between correctness and coverage.
    Accuracy (الصحة): correctness across retrieved-or-relevant history items.

    Reads evaluable ``chat_entries`` from ``db_path`` and returns the same
    micro-averaged structure as ``evaluate_batch``, plus
    ``documents_evaluated``.
    """
    results = load_history_results(
        db_path,
        session_id=session_id,
        retrieved_key=retrieved_key,
        relevant_key=relevant_key,
        apply_precision_filters=apply_precision_filters,
    )
    evaluation = evaluate_batch(results, normalize=normalize)
    evaluation["documents_evaluated"] = len(results)
    return evaluation


def analyze_history_false_positives(
    db_path: str | Path = DEFAULT_HISTORY_DB_PATH,
    *,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Return categorized false positives from the unfiltered history baseline."""
    baseline_records = load_history_evaluation_records(
        db_path,
        session_id=session_id,
        apply_precision_filters=False,
    )
    filtered_records = load_history_evaluation_records(
        db_path,
        session_id=session_id,
        apply_precision_filters=True,
    )
    filtered_by_id = {record["entry_id"]: record for record in filtered_records}

    false_positives: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}

    for record in baseline_records:
        normalized_relevant = {
            _normalize_financial_item(item)
            for item in record["relevant"]
        }
        filtered_record = filtered_by_id.get(record["entry_id"], {})
        filtered_retrieved = {
            _normalize_financial_item(item)
            for item in filtered_record.get("retrieved", set())
        }

        for rank, item in enumerate(sorted(record["retrieved"]), start=1):
            normalized_item = _normalize_financial_item(item)
            if normalized_item in normalized_relevant:
                continue

            decision = _grounding_decision(
                item,
                set(record.get("source_items") or set()),
                str(record.get("question") or ""),
            )
            category = str(decision["category"])
            if normalized_item in filtered_retrieved:
                category = "normalization_or_support_context_mismatch"
                decision["reason"] = (
                    "Legacy support missed this item, but strict grounding accepts it "
                    "after scale/row normalization."
                )

            category_counts[category] = category_counts.get(category, 0) + 1
            false_positives.append(
                {
                    "entry_id": record["entry_id"],
                    "question": record["question"],
                    "rank": rank,
                    "similarity_or_rank_score": decision["similarity_or_rank_score"],
                    "support_score": decision["support_score"],
                    "query_score": decision["query_score"],
                    "category": category,
                    "retrieved_item": item,
                    "source_chunk": decision["source_chunk"],
                    "reason": decision["reason"],
                }
            )

    return {
        "total_false_positives": len(false_positives),
        "category_counts": dict(sorted(category_counts.items())),
        "false_positives": false_positives,
        "score_note": (
            "Historical chat rows did not save raw retrieval similarity scores; "
            "similarity_or_rank_score is null for those rows. support_score and "
            "query_score are deterministic post-extraction rerank scores."
        ),
    }


class _MissingValue:
    pass


_MISSING = _MissingValue()


def _has_chat_entries_table(connection: sqlite3.Connection) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chat_entries'"
    ).fetchone()
    return row is not None


def _field_value(entry: dict[str, Any], preferred_key: str | None, aliases: tuple[str, ...]) -> Any:
    if preferred_key is not None:
        return entry.get(preferred_key, _MISSING)
    for key in aliases:
        if key in entry:
            return entry[key]
    return _MISSING


def _to_item_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, dict):
        return {f"{key}: {_stringify_item(item)}" for key, item in value.items()}
    if isinstance(value, (list, set, tuple)):
        items: set[str] = set()
        for item in value:
            items.update(_to_item_set(item))
        return items
    text = _stringify_item(value)
    if _looks_like_text_blob(text):
        extracted = extract_financial_items(text)
        if extracted:
            return extracted
    return {text} if text else set()


def _to_raw_text_list(value: Any) -> list[str]:
    if value is None or value is _MISSING:
        return []
    if isinstance(value, dict):
        return [json.dumps(value, sort_keys=True, ensure_ascii=False)]
    if isinstance(value, (list, set, tuple)):
        texts: list[str] = []
        for item in value:
            texts.extend(_to_raw_text_list(item))
        return texts
    text = _stringify_item(value)
    return [text] if text else []


def _dedupe_financial_items(items: set[str]) -> set[str]:
    """Collapse normalized duplicate answer items while preserving readable text."""
    deduped: dict[str, str] = {}
    for item in sorted(items, key=lambda value: (len(value), value)):
        key = _normalize_financial_item(item)
        if key not in deduped:
            deduped[key] = item
    return set(deduped.values())


def extract_financial_items(text: str) -> set[str]:
    """Extract financial item candidates from answer/source text.

    Precision (الدقة), Recall (الاستدعاء), F1-Score (مؤشر التوازن), and
    Accuracy (الصحة) can be computed from chat-history rows by deriving
    numeric financial clauses from saved answers and source chunks.
    """
    items: set[str] = set()
    cleaned = str(text).replace("\u202f", " ").replace("\xa0", " ")
    fragments = re.split(r"[\n\r]+|(?:\s+[•*-]\s+)|(?:\s+\d+[.)]\s+)|\|", cleaned)
    for fragment in fragments:
        fragment = re.sub(r"^\s*[-•*]\s*", "", fragment).strip()
        fragment = re.sub(r"\s+", " ", fragment)
        if not fragment or not re.search(r"\d|[٠-٩]", fragment):
            continue
        fragment = re.sub(r"^\[Source:[^\]]+\]\s*", "", fragment).strip()
        if len(fragment) > 240:
            fragment = fragment[:240].rsplit(" ", 1)[0].strip()
        if fragment:
            items.add(fragment)
    return items


def _stringify_item(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value).strip()


def _is_supported_by_source_items(item: str, source_items: set[str]) -> bool:
    item_numbers = _legacy_number_tokens(item)
    if not item_numbers:
        return False
    item_words = _legacy_word_tokens(item)
    for source_item in source_items:
        source_numbers = _legacy_number_tokens(source_item)
        if not item_numbers.issubset(source_numbers):
            continue
        source_words = _legacy_word_tokens(source_item)
        if item_words and source_words and item_words & source_words:
            return True
        if len(item_numbers) >= 2:
            return True
    return False


def _grounding_decision(item: str, source_items: set[str], query: str = "") -> dict[str, Any]:
    """Score one extracted item against source evidence and query intent."""
    material_numbers = _material_number_tokens(item)
    if _is_no_data_response(item) or not material_numbers:
        return _decision(
            item,
            accepted=False,
            category="non_financial_or_no_data_response",
            reason="The item does not contain a material financial figure to score.",
            material_numbers=material_numbers,
        )

    item_words = _meaningful_word_tokens(item)
    best_source = ""
    best_source_words: set[str] = set()
    best_number_overlap = 0.0
    best_word_overlap = 0.0
    all_source_numbers: set[str] = set()

    for source_item in source_items:
        source_numbers = _material_number_tokens(source_item)
        all_source_numbers.update(source_numbers)
        if not source_numbers:
            continue

        number_overlap = len(material_numbers & source_numbers) / len(material_numbers)
        source_words = _meaningful_word_tokens(source_item)
        if item_words:
            word_overlap = len(item_words & source_words) / len(item_words)
        else:
            word_overlap = 0.0

        score = (number_overlap, word_overlap, len(source_item))
        best_score = (best_number_overlap, best_word_overlap, len(best_source))
        if score > best_score:
            best_source = source_item
            best_source_words = source_words
            best_number_overlap = number_overlap
            best_word_overlap = word_overlap

    support_score = round((0.75 * best_number_overlap) + (0.25 * min(best_word_overlap * 2, 1.0)), 3)

    if best_number_overlap < 1.0:
        if material_numbers and material_numbers.issubset(all_source_numbers):
            category = "cross_row_or_blended_figure"
            reason = (
                "The item's numbers appear in the source set, but not in one exact row/chunk span."
            )
        else:
            category = "ungrounded_or_hallucinated_figure"
            reason = "At least one material number is absent from the retrieved source chunks."
        return _decision(
            item,
            accepted=False,
            category=category,
            reason=reason,
            source_chunk=best_source,
            support_score=support_score,
            query_score=0.0,
            material_numbers=material_numbers,
        )

    if item_words and best_source_words and not (item_words & best_source_words):
        return _decision(
            item,
            accepted=False,
            category="wrong_label_or_neighboring_row",
            reason="The number exists, but the source row label does not match the answer item.",
            source_chunk=best_source,
            support_score=support_score,
            query_score=0.0,
            material_numbers=material_numbers,
        )

    query_score = _query_relevance_score(query, item_words | best_source_words)
    if query.strip() and query_score <= 0:
        return _decision(
            item,
            accepted=False,
            category="query_intent_mismatch",
            reason="The item is grounded, but it does not match the question's target field.",
            source_chunk=best_source,
            support_score=support_score,
            query_score=query_score,
            material_numbers=material_numbers,
        )

    return _decision(
        item,
        accepted=True,
        category="grounded_relevant_item",
        reason="The item is directly grounded in one source row/chunk and matches the query intent.",
        source_chunk=best_source,
        support_score=support_score,
        query_score=query_score,
        material_numbers=material_numbers,
    )


def _decision(
    item: str,
    *,
    accepted: bool,
    category: str,
    reason: str,
    source_chunk: str = "",
    support_score: float = 0.0,
    query_score: float = 0.0,
    material_numbers: set[str] | None = None,
) -> dict[str, Any]:
    return {
        "accepted": accepted,
        "category": category,
        "retrieved_item": item,
        "source_chunk": source_chunk,
        "similarity_or_rank_score": None,
        "support_score": support_score,
        "query_score": query_score,
        "material_numbers": sorted(material_numbers or set()),
        "reason": reason,
    }


def _number_tokens(text: str) -> set[str]:
    normalized = _normalize_value(_prepare_financial_text(text))
    return {token for token in normalized.split() if token.startswith(("num_", "pct_"))}


def _legacy_number_tokens(text: str) -> set[str]:
    normalized = _normalize_value(_legacy_prepare_financial_text(text))
    return {token for token in normalized.split() if token.startswith(("num_", "pct_"))}


def _material_number_tokens(text: str) -> set[str]:
    prepared = _strip_scale_denominators(_prepare_financial_text(text))
    normalized = _normalize_value(prepared)
    return {token for token in normalized.split() if token.startswith(("num_", "pct_"))}


def _word_tokens(text: str) -> set[str]:
    prepared = _prepare_financial_text(text)
    prepared = _NUMBER_RE.sub(" ", prepared)
    tokens = set(re.findall(r"[a-z\u0600-\u06ff]{2,}", prepared))
    return tokens - {"sar", "usd", "eur", "gbp", "من", "في", "على", "هو", "هي", "مع"}


def _legacy_word_tokens(text: str) -> set[str]:
    prepared = _legacy_prepare_financial_text(text)
    prepared = _NUMBER_RE.sub(" ", prepared)
    tokens = set(re.findall(r"[a-z\u0600-\u06ff]{2,}", prepared))
    return tokens - {"sar", "usd", "eur", "gbp", "من", "في", "على", "هو", "هي", "مع"}


def _meaningful_word_tokens(text: str) -> set[str]:
    return {
        token
        for token in _word_tokens(text)
        if token not in _STOPWORDS and len(token) > 1
    }


def _query_relevance_score(query: str, candidate_words: set[str]) -> float:
    query_words = _meaningful_word_tokens(query)
    if not query_words:
        return 1.0
    if not candidate_words:
        return 0.0
    overlap = query_words & candidate_words
    if not overlap:
        return 0.0
    return round(min(len(overlap) / min(len(query_words), 4), 1.0), 3)


def _strip_scale_denominators(text: str) -> str:
    return _SCALE_DENOMINATOR_RE.sub(r"\1", text)


def _is_no_data_response(text: str) -> bool:
    return bool(_NO_DATA_RESPONSE_RE.search(str(text)))


def _looks_like_text_blob(text: str) -> bool:
    return (
        len(text) > 80
        or "\n" in text
        or "|" in text
        or "  " in text
        or text.lstrip().startswith(("-", "*", "•"))
    )


def _normalize_financial_item(item: str) -> str:
    text = _prepare_financial_text(item)
    if ":" in text:
        label, value = text.split(":", 1)
        return f"{_normalize_label(label)}: {_normalize_value(value)}"
    return _normalize_value(text)


def _prepare_financial_text(text: Any) -> str:
    normalized = str(text).translate(_ARABIC_DIGITS).lower().strip()
    normalized = _ARABIC_DIACRITICS_RE.sub("", normalized)
    normalized = re.sub(r"\bunnamed\s*:?\s*\d+\b", " ", normalized)
    for old, new in _CURRENCY_REPLACEMENTS.items():
        normalized = normalized.replace(old, new)
    return normalized


def _legacy_prepare_financial_text(text: Any) -> str:
    normalized = str(text).translate(_ARABIC_DIGITS).lower().strip()
    for old, new in _CURRENCY_REPLACEMENTS.items():
        if old == "٪":
            continue
        normalized = normalized.replace(old, new)
    return normalized


def _normalize_label(text: str) -> str:
    text = re.sub(r"[^a-z0-9\u0600-\u06ff]+", " ", text)
    return " ".join(text.split())


def _normalize_value(text: str) -> str:
    text = _NUMBER_RE.sub(lambda match: _canonical_number_token(match.group(0)), text)
    text = re.sub(r"[^a-z0-9_\u0600-\u06ff]+", " ", text)
    tokens = [token for token in text.split() if token]
    return " ".join(sorted(tokens))


def _canonical_number_token(raw_value: str) -> str:
    raw = raw_value.strip().lower()
    negative = raw.startswith("-") or (raw.startswith("(") and raw.endswith(")"))
    unit_match = re.search(r"(percent|percentage|million|mn|m|billion|bn|b|thousand|k|%)\s*$", raw)
    unit = unit_match.group(1) if unit_match else ""
    number_text = re.sub(r"[^0-9.]", "", raw)
    if not number_text:
        return raw_value

    try:
        number = Decimal(number_text)
    except InvalidOperation:
        return raw_value

    if negative:
        number *= Decimal("-1")

    if unit in _UNIT_FACTORS:
        number *= _UNIT_FACTORS[unit]

    prefix = "pct" if unit in {"%", "percent", "percentage"} else "num"
    formatted = format(number.normalize(), "f")
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    safe = formatted.replace("-", "neg").replace(".", "p")
    return f" {prefix}_{safe} "


if __name__ == "__main__":
    current_pairs = load_history_results(apply_precision_filters=True)
    current = evaluate_batch(current_pairs, normalize=True)
    output: dict[str, Any] = {
        "source": "src/chat_history.db",
        "documents_evaluated": len(current_pairs),
        "current_overall_evaluation": current,
    }
    if not current_pairs:
        output["message"] = (
            "No evaluable chat-history rows found. New rows must include both "
            "'retrieved' (answer/extracted items) and 'relevant' or 'source_texts' "
            "(source/ground-truth items)."
        )

    print(json.dumps(output, indent=2, ensure_ascii=False))
    print_overall_evaluation(output)
