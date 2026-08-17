import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_insight_lite.modules.processor import csv_to_documents_optimized, load_documents_fastest
from finance_insight_lite.modules.rag_agent import CRAGRetriever, SelfRefiningAnswerEngine
from finance_insight_lite.modules.structured_tables import (
    is_small_table,
    render_full_table_document,
    small_table_documents,
)
from finance_insight_lite.modules.verctor_store import _load_or_create_chunks
from finance_insight_lite.modules.workflow_coordinator import WorkflowCoordinator


class FakeRetriever:
    def __init__(self, full_table_documents=None, relevant_documents=None):
        self.full_table_documents = full_table_documents or []
        self.relevant_documents = relevant_documents or []
        self.crag_called = False

    def get_full_table_documents(self, query=""):
        return self.full_table_documents

    def get_relevant_documents(self, query, k=None):
        self.crag_called = True
        return [{"document": document, "relevant": True} for document in self.relevant_documents]


class FakeLLM:
    def with_structured_output(self, _schema):
        return self


class CapturingLLM:
    def __init__(self, content="Captured response."):
        self.content = content
        self.last_messages = None

    def with_structured_output(self, _schema):
        return self

    def invoke(self, messages):
        self.last_messages = messages

        class Response:
            def __init__(self, content):
                self.content = content

        return Response(self.content)


class StructuredTableRetrievalTests(unittest.TestCase):
    def test_csv_loader_emits_one_document_per_row_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "ledger.csv"
            pd.DataFrame({
                "entry_id": ["E-101", "E-102"],
                "amount_sar": [1000, 2000],
            }).to_csv(csv_path, index=False)

            documents = csv_to_documents_optimized(str(csv_path))

        self.assertEqual(len(documents), 2)
        self.assertEqual([doc.metadata["row"] for doc in documents], [1, 2])
        self.assertTrue(all(doc.metadata["table_row"] for doc in documents))
        self.assertIn("entry_id: E-101", documents[0].page_content)

    def test_fast_loader_adds_source_file_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "ledger.csv"
            pd.DataFrame({
                "entry_id": ["E-101"],
                "amount_sar": [1000],
            }).to_csv(csv_path, index=False)

            result = load_documents_fastest(str(csv_path), use_cache=False)

        self.assertEqual(result["documents"][0].metadata["source_file"], "ledger.csv")

    def test_tabular_rows_are_not_resplit_or_overlapped_in_vector_chunks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "ledger.csv"
            pd.DataFrame({
                "entry_id": ["ROW-001", "ROW-002", "ROW-003"],
                "amount_sar": [1000, 2000, 3000],
            }).to_csv(csv_path, index=False)
            documents = csv_to_documents_optimized(str(csv_path))

            chunks = _load_or_create_chunks(documents, Path(tmpdir) / "chunks.pkl")

        self.assertEqual(len(chunks), 3)
        for marker in ["ROW-001", "ROW-002", "ROW-003"]:
            occurrences = sum(chunk.page_content.count(marker) for chunk in chunks)
            self.assertEqual(occurrences, 1)

    def test_small_excel_file_renders_full_table_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path = Path(tmpdir) / "pipeline.xlsx"
            pd.DataFrame({
                "deal_id": ["DEAL-201", "DEAL-202"],
                "value_sar": [75000, 90000],
            }).to_excel(xlsx_path, index=False)

            self.assertTrue(is_small_table(str(xlsx_path)))
            document = render_full_table_document(str(xlsx_path))

        self.assertTrue(document.metadata["table_full_context"])
        self.assertEqual(document.metadata["source_file"], "pipeline.xlsx")
        self.assertIn("DEAL-201", document.page_content)
        self.assertIn("DEAL-202", document.page_content)

    def test_large_csv_does_not_get_full_table_bypass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "large.csv"
            pd.DataFrame({
                "row_id": [f"ROW-{i:03d}" for i in range(501)],
                "amount_sar": list(range(501)),
            }).to_csv(csv_path, index=False)

            self.assertFalse(is_small_table(str(csv_path)))
            self.assertEqual(small_table_documents([str(csv_path)]), [])

    def test_workflow_routes_small_tables_before_crag(self):
        full_table = Document(
            page_content="Source file: ledger.csv\nCSV:\nentry_id,amount_sar\nE-101,1000",
            metadata={"source": "ledger.csv", "table_full_context": True},
        )
        normal_doc = Document(page_content="fallback", metadata={})
        retriever = FakeRetriever(
            full_table_documents=[full_table],
            relevant_documents=[normal_doc],
        )
        coordinator = WorkflowCoordinator(fast_llm=FakeLLM(), crag_retriever=retriever)

        routed = coordinator.route("reconcile ledger")

        self.assertEqual(routed["instruction"].action, "ANSWER_FROM_FULL_TABLE")
        self.assertEqual(routed["documents"], [full_table])
        self.assertFalse(retriever.crag_called)

    def test_workflow_falls_back_to_crag_without_small_tables(self):
        normal_doc = Document(page_content="fallback", metadata={})
        retriever = FakeRetriever(relevant_documents=[normal_doc])
        coordinator = WorkflowCoordinator(fast_llm=FakeLLM(), crag_retriever=retriever)

        routed = coordinator.route("summarize the filing")

        self.assertEqual(routed["instruction"].action, "ANSWER_FROM_CONTEXT")
        self.assertEqual(routed["documents"], [normal_doc])
        self.assertTrue(retriever.crag_called)

    def test_crag_dedupes_retrieved_chunks_and_scores_by_content(self):
        docs = [
            Document(page_content="amount_sar: 1000", metadata={"row": 1}),
            Document(page_content=" amount_sar: 1000 ", metadata={"row": 1}),
            Document(page_content="amount_sar: 2000", metadata={"row": 2}),
        ]

        deduped_docs, deduped_scores = CRAGRetriever._dedupe_docs(docs, [0.9, 0.8, 0.7])

        self.assertEqual([doc.page_content.strip() for doc in deduped_docs], [
            "amount_sar: 1000",
            "amount_sar: 2000",
        ])
        self.assertEqual(deduped_scores, [0.9, 0.7])

    def test_unverified_numbers_are_removed_from_final_answer_text(self):
        answer = "Open pipeline is 165,000 SAR.\nDEAL-206 duplicate is 25,000 SAR."
        verification = {
            "number_checks": [
                {
                    "number_in_answer": "25,000",
                    "row_label_in_source": "NOT_FOUND_IN_SOURCES",
                    "matches_question_intent": False,
                }
            ]
        }

        cleaned = SelfRefiningAnswerEngine._remove_unverified_numbers(answer, verification)

        self.assertIn("Open pipeline is 165,000 SAR.", cleaned)
        self.assertNotIn("DEAL-206 duplicate", cleaned)
        self.assertIn("The figure 25,000 was not found in the provided data.", cleaned)

    def test_english_query_sets_english_response_language_despite_arabic_context(self):
        llm = CapturingLLM()
        engine = SelfRefiningAnswerEngine(llm)

        engine._generate_initial(
            query="what is the net income?",
            context="الحقائق: صافي الدخل هو 109,520 مليون ريال.",
            chat_history=[{"question": "ما هو صافي الدخل؟", "answer": "الإجابة السابقة بالعربية."}],
        )

        prompt_text = "\n".join(str(message.content) for message in llm.last_messages)
        self.assertIn("TARGET RESPONSE LANGUAGE: English", prompt_text)
        self.assertIn("what is the net income?", prompt_text)

    def test_refinement_uses_query_language_not_previous_answer_language(self):
        llm = CapturingLLM()
        engine = SelfRefiningAnswerEngine(llm)

        engine._refine(
            query="what is the net income?",
            context="Net income was SAR 109,520 million.",
            previous_answer="الحقائق: صافي الدخل هو 109,520 مليون ريال.",
            verification={"rating": 5, "missing_refs": [], "notes": "Wrong language."},
        )

        prompt_text = "\n".join(str(message.content) for message in llm.last_messages)
        self.assertIn("TARGET RESPONSE LANGUAGE: English", prompt_text)
        self.assertIn("previous answer used a different language", prompt_text)


if __name__ == "__main__":
    unittest.main()
