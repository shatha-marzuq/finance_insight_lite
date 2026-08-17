from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from evaluation.metrics import (
    accuracy,
    analyze_history_false_positives,
    analyze_retrieval_errors,
    evaluate,
    evaluate_batch,
    evaluate_history_db,
    extract_financial_items,
    f1_score,
    load_history_results,
    load_jsonl_results,
    precision,
    recall,
)


CHAT_HISTORY_EXPORT_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "chat_history_export.jsonl"


def test_core_metrics_normal_case() -> None:
    assert precision(true_positives=3, false_positives=1) == pytest.approx(0.75)
    assert recall(true_positives=3, false_negatives=2) == pytest.approx(0.6)
    assert f1_score(0.75, 0.6) == pytest.approx(2 * (0.75 * 0.6) / (0.75 + 0.6))
    assert accuracy(true_positives=3, false_positives=1, false_negatives=2) == pytest.approx(3 / 6)


def test_evaluate_computes_set_based_counts_and_metrics() -> None:
    result = evaluate(
        retrieved={"revenue: 100m", "net income: 15m", "ebitda: 30m"},
        relevant={"revenue: 100m", "net income: 15m", "cash flow: 22m"},
    )

    assert result["true_positives"] == 2
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1_score"] == pytest.approx(2 / 3)
    assert result["accuracy"] == pytest.approx(2 / 4)


def test_evaluate_normalizes_equivalent_financial_values() -> None:
    retrieved = {
        "revenue: 100m SAR",
        "net income: 15m SAR",
        "cash flow: SAR 22 million",
    }
    relevant = {
        "revenue: 100m SAR",
        "net income: 15m SAR",
        "cash flow: 22,000,000 SAR",
    }

    before = evaluate(retrieved, relevant, normalize=False)
    after = evaluate(retrieved, relevant)

    assert before["true_positives"] == 2
    assert before["false_positives"] == 1
    assert before["false_negatives"] == 1
    assert after["true_positives"] == 3
    assert after["false_positives"] == 0
    assert after["false_negatives"] == 0
    assert after["precision"] == pytest.approx(1.0)
    assert after["recall"] == pytest.approx(1.0)
    assert after["f1_score"] == pytest.approx(1.0)


def test_analyze_retrieval_errors_categorizes_normalization_issue() -> None:
    analysis = analyze_retrieval_errors(
        retrieved={"cash flow: SAR 22 million"},
        relevant={"cash flow: 22,000,000 SAR"},
    )

    assert analysis["exact_false_positives"] == ["cash flow: SAR 22 million"]
    assert analysis["exact_false_negatives"] == ["cash flow: 22,000,000 SAR"]
    assert analysis["likely_cause"] == "matching/normalization"
    assert analysis["categorized_errors"][0]["category"] == "matching/normalization"


def test_extract_financial_items_from_answer_or_source_text() -> None:
    items = extract_financial_items(
        "**Facts**\n"
        "- Revenue was 100m SAR.\n"
        "- Cash flow was 22,000,000 SAR.\n"
        "No numeric detail here."
    )

    assert "Revenue was 100m SAR." in items
    assert "Cash flow was 22,000,000 SAR." in items
    assert all("No numeric detail" not in item for item in items)


def test_zero_division_edge_cases_return_zero() -> None:
    assert precision(true_positives=0, false_positives=0) == 0.0
    assert recall(true_positives=0, false_negatives=0) == 0.0
    assert f1_score(precision_value=0.0, recall_value=0.0) == 0.0
    assert accuracy(true_positives=0, false_positives=0, false_negatives=0) == 0.0

    empty_result = evaluate(retrieved=set(), relevant=set())

    assert empty_result == {
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "accuracy": 0.0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
    }


def test_retrieved_empty_relevant_non_empty() -> None:
    result = evaluate(retrieved=set(), relevant={"revenue: 100m", "net income: 15m"})

    assert result["true_positives"] == 0
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 2
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1_score"] == 0.0
    assert result["accuracy"] == 0.0


def test_evaluate_batch_micro_averages_across_documents() -> None:
    result = evaluate_batch(
        [
            ({"revenue", "net income"}, {"revenue", "net income", "capex"}),
            ({"cash flow", "ebitda"}, {"cash flow"}),
            (set(), {"debt"}),
        ]
    )

    assert result["true_positives"] == 3
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 2
    assert result["precision"] == pytest.approx(3 / 4)
    assert result["recall"] == pytest.approx(3 / 5)
    assert result["f1_score"] == pytest.approx(2 * ((3 / 4) * (3 / 5)) / ((3 / 4) + (3 / 5)))
    assert result["accuracy"] == pytest.approx(3 / 6)
    assert len(result["per_document"]) == 3


def test_load_jsonl_results_reads_exported_chat_history_fixture() -> None:
    pairs = load_jsonl_results(CHAT_HISTORY_EXPORT_PATH)

    expected_rows = sum(1 for line in CHAT_HISTORY_EXPORT_PATH.read_text(encoding="utf-8").splitlines() if line.strip())
    assert len(pairs) == expected_rows
    assert pairs[0][1] == {"test_2_budget_actuals.xlsx", "test_3_revops_data.csv"}


def test_evaluate_history_db_micro_averages_evaluable_entries(tmp_path) -> None:
    db_path = tmp_path / "chat_history.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE chat_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                entry_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        entries = [
            {
                "extracted_items": ["revenue", "net income"],
                "expected_items": ["revenue", "net income", "capex"],
            },
            {
                "extracted_items": ["cash flow", "ebitda"],
                "expected_items": ["cash flow"],
            },
            {"answer": "chat entry without relevant items"},
        ]
        for index, entry in enumerate(entries):
            connection.execute(
                "INSERT INTO chat_entries (session_id, entry_json, created_at) VALUES (?, ?, ?)",
                ("session-a", json.dumps(entry), float(index)),
            )

    result = evaluate_history_db(db_path)

    assert result["documents_evaluated"] == 2
    assert result["true_positives"] == 3
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["precision"] == pytest.approx(3 / 4)
    assert result["recall"] == pytest.approx(3 / 4)
    assert result["f1_score"] == pytest.approx(3 / 4)
    assert result["accuracy"] == pytest.approx(3 / 5)


def test_load_history_results_derives_items_from_answer_and_source_texts(tmp_path) -> None:
    db_path = tmp_path / "chat_history.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE chat_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                entry_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO chat_entries (session_id, entry_json, created_at) VALUES (?, ?, ?)",
            (
                "session-a",
                json.dumps(
                    {
                        "question": "show revenue and cash flow",
                        "answer": "**Facts**\n- Revenue was 100m SAR.\n- Cash flow was SAR 22 million.",
                        "source_texts": [
                            "[Source: report.xlsx]\nRevenue was 100m SAR | Cash flow was 22,000,000 SAR"
                        ],
                    }
                ),
                1.0,
            ),
        )

    pairs = load_history_results(db_path)

    assert pairs == [
        (
            {"Revenue was 100m SAR.", "Cash flow was SAR 22 million."},
            {"Revenue was 100m SAR.", "Cash flow was SAR 22 million."},
        )
    ]


def test_load_history_results_treats_relevant_source_text_copy_as_support_context(tmp_path) -> None:
    db_path = tmp_path / "chat_history.db"
    source_texts = ["[Source: report.xlsx]\nRevenue was 100m SAR | Cash flow was 22,000,000 SAR"]
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE chat_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                entry_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO chat_entries (session_id, entry_json, created_at) VALUES (?, ?, ?)",
            (
                "session-a",
                json.dumps(
                    {
                        "answer": "- Revenue was 100m SAR.\n- Unsupported margin was 9%.",
                        "relevant": source_texts,
                        "source_texts": source_texts,
                    }
                ),
                1.0,
            ),
        )

    pairs = load_history_results(db_path)

    assert pairs == [
        (
            {"Revenue was 100m SAR."},
            {"Revenue was 100m SAR."},
        )
    ]

    baseline_pairs = load_history_results(db_path, apply_precision_filters=False)

    assert baseline_pairs == [
        (
            {"Revenue was 100m SAR.", "Unsupported margin was 9%."},
            {"Revenue was 100m SAR."},
        )
    ]


def test_precision_filter_accepts_scale_denominator_but_rejects_blended_rows(tmp_path) -> None:
    db_path = tmp_path / "chat_history.db"
    source_texts = [
        "[Source: impact.xlsx]\n"
        "المؤشر,المعدل (1-4),% من المشاركين\n"
        "العائد مقابل الوقت والجهد المستثمر (ROI),3.71 من 4, 100%\n"
        "اكتشاف مواهب جديدة,57%\n"
        "تذكر السمات الخمس الأولى,86%"
    ]
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE chat_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                entry_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO chat_entries (session_id, entry_json, created_at) VALUES (?, ?, ?)",
            (
                "session-a",
                json.dumps(
                    {
                        "question": "ما هو مؤشر ROI؟",
                        "answer": (
                            "- مؤشر ROI هو 3.71 من 4 مع 100% من المشاركين.\n"
                            "- 57% اكتشفوا مواهب جديدة و86% يتذكرون السمات."
                        ),
                        "relevant": source_texts,
                        "source_texts": source_texts,
                    }
                ),
                1.0,
            ),
        )

    pairs = load_history_results(db_path)

    assert pairs == [
        (
            {"مؤشر ROI هو 3.71 من 4 مع 100% من المشاركين."},
            {"مؤشر ROI هو 3.71 من 4 مع 100% من المشاركين."},
        )
    ]

    breakdown = analyze_history_false_positives(db_path)

    assert breakdown["category_counts"]["cross_row_or_blended_figure"] == 1


def test_precision_filter_does_not_replace_relevant_with_filtered_retrieved(tmp_path) -> None:
    db_path = tmp_path / "chat_history.db"
    source_texts = [
        "[Source: impact.xlsx]\n"
        "المؤشر,المعدل (1-4),% من المشاركين\n"
        "العائد مقابل الوقت والجهد المستثمر (ROI),3.71,100%"
    ]
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE chat_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                entry_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO chat_entries (session_id, entry_json, created_at) VALUES (?, ?, ?)",
            (
                "session-a",
                json.dumps(
                    {
                        "question": "ما هو مؤشر ROI؟",
                        "answer": "- مؤشر ROI هو 3.71 من 4 مع 100% من المشاركين.",
                        "relevant": source_texts,
                        "source_texts": source_texts,
                    }
                ),
                1.0,
            ),
        )

    pairs = load_history_results(db_path)

    assert pairs == [
        (
            {"مؤشر ROI هو 3.71 من 4 مع 100% من المشاركين."},
            set(),
        )
    ]


def test_load_jsonl_results_skips_rows_without_reference_side(tmp_path) -> None:
    fixture_path = tmp_path / "heldout.jsonl"
    fixture_path.write_text(
        json.dumps(
            {
                "retrieved": ["revenue: SAR 100m", "margin: 9%"],
                "relevant": ["revenue: SAR 100m"],
            }
        )
        + "\n"
        + json.dumps({"retrieved": ["not evaluable without relevant"]})
        + "\n",
        encoding="utf-8",
    )

    pairs = load_jsonl_results(fixture_path)

    assert pairs == [
        (
            {"revenue: SAR 100m", "margin: 9%"},
            {"revenue: SAR 100m"},
        )
    ]
