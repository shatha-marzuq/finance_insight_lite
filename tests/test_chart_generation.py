"""Focused tests for chart intent routing and Plotly generation."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_insight_lite.modules.rag_agent import (
    AggregationIntent,
    ChartDecision,
    ChartGenerator,
    ChartIntent,
    ChartIntentDetector,
    ChartTypeSelector,
    DeterministicContextAnalyzer,
    FinancialDataExtractor,
    FinancialRAGAgent,
    SelfRAGVerifier,
    SelfRefiningAnswerEngine,
)


class FakeVectorDB:
    def __init__(self, documents):
        self.documents = documents
        self.queries = []
        self.index = type("FakeIndex", (), {"ntotal": len(documents)})()

    def similarity_search(self, query, k):
        self.queries.append((query, k))
        return self.documents[:k]


class FailIfInvokedLLM:
    def invoke(self, _messages):
        raise AssertionError("Structured improvement rows should not require the LLM")


class FakeStructuredChartLLM:
    def __init__(self, decision):
        self.decision = decision
        self.last_prompt = None
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        self.last_prompt = "\n".join(str(message.content) for message in messages)
        return self.decision


class FakeChartLLM:
    def __init__(self, decision):
        self.structured = FakeStructuredChartLLM(decision)

    def with_structured_output(self, schema):
        self.schema = schema
        return self.structured


class FailingStructuredLLM:
    def invoke(self, _messages):
        raise RuntimeError("structured llm unavailable")


class FailingStructuredChatLLM:
    def with_structured_output(self, _schema):
        return FailingStructuredLLM()


class FakeTextResponse:
    def __init__(self, content):
        self.content = content


class CapturingTextLLM:
    def __init__(self):
        self.last_messages = None

    def with_structured_output(self, _schema):
        return FakeStructuredChartLLM(ChartIntent(wants_chart=False))

    def invoke(self, messages):
        self.last_messages = messages
        return FakeTextResponse("Highest stage is Qualified.")


class ChartGenerationTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {"label": ["Q1", "Q2", "Q3"], "value": [100.0, 125.5, 140.0]}
        )
        self.agent = FinancialRAGAgent.__new__(FinancialRAGAgent)

    def test_arabic_chart_request_is_recognized(self):
        self.assertTrue(
            self.agent._needs_chart("ابي رسم بياني يوضح لي نسبة التحسن")
        )

    def test_non_chart_arabic_question_is_not_recognized_as_chart(self):
        self.assertFalse(self.agent._needs_chart("ما هي نسبة التحسن؟"))

    def test_chart_type_selector_uses_structured_llm_for_arabic_intent(self):
        fast_llm = FakeChartLLM(
            ChartDecision(chart_type="line", reason="Arabic query asks for time trend")
        )
        selector = ChartTypeSelector(fast_llm)

        decision = selector.select("الاتجاه خلال الأشهر الماضية", self.df)

        self.assertEqual(decision.chart_type, "line")
        self.assertIn("Question: الاتجاه خلال الأشهر الماضية", fast_llm.structured.last_prompt)
        self.assertIn("Number of data points: 3", fast_llm.structured.last_prompt)

    def test_chart_intent_detector_handles_failing_arabic_draw_query(self):
        fast_llm = FakeChartLLM(ChartIntent(wants_chart=True))
        detector = ChartIntentDetector(
            fast_llm,
            fallback_keywords=FinancialRAGAgent.VIZ_KEYWORDS,
        )

        self.assertTrue(detector.detect("ارسم لي مقارنة بين قيم الصفقات حسب المرحلة"))

    def test_chart_intent_detector_rejects_plain_arabic_fact_question(self):
        fast_llm = FakeChartLLM(ChartIntent(wants_chart=False))
        detector = ChartIntentDetector(
            fast_llm,
            fallback_keywords=FinancialRAGAgent.VIZ_KEYWORDS,
        )

        self.assertFalse(detector.detect("كم إجمالي قيمة الصفقات؟"))

    def test_chart_intent_fallback_recognizes_arabic_draw(self):
        detector = ChartIntentDetector(
            FailingStructuredChatLLM(),
            fallback_keywords=FinancialRAGAgent.VIZ_KEYWORDS,
        )

        self.assertTrue(detector.detect("ارسم لي مقارنة بين قيم الصفقات حسب المرحلة"))

    def test_chart_intent_detector_rejects_plain_english_compare(self):
        fast_llm = FakeChartLLM(ChartIntent(wants_chart=False))
        detector = ChartIntentDetector(
            fast_llm,
            fallback_keywords=FinancialRAGAgent.VIZ_KEYWORDS,
        )

        self.assertFalse(detector.detect("compare deal values across pipeline stages"))

    def test_fallback_chart_types_are_selected_when_llm_unavailable(self):
        cases = {
            "ارسم مخططاً دائرياً لتوزيع المصروفات": "pie",
            "ارسم مخططاً خطياً للتطور الشهري": "line",
            "ارسم مخطط أعمدة للمقارنة": "bar",
            "ارسم مخططاً مساحياً للتدفق النقدي": "area",
        }

        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertTrue(self.agent._needs_chart(query))
                self.assertEqual(
                    self.agent._fallback_chart_type(query, self.df), expected
                )

    def test_improvement_percentage_uses_bar_chart(self):
        query = "ابي رسم بياني يوضح لي نسبة التحسن"

        self.assertEqual(self.agent._fallback_chart_type(query, self.df), "bar")

    def test_improvement_extraction_keeps_only_comparable_percentages(self):
        documents = [
            Document(page_content=(
                "البُعد المقاس: معرفتي بمفهوم مواطن القوّة | الفرق: 1.95 | "
                "نسبة التحسّن: +120%"
            )),
            Document(page_content=(
                "البُعد المقاس: بداية استشعاري لمواهبي | الفرق: 1.17 | "
                "نسبة التحسّن: +55%"
            )),
            Document(page_content=(
                "المؤشر: الرضا العام | المعدّل (1-4): 3.57"
            )),
        ]
        vector_db = FakeVectorDB(documents)
        extractor = FinancialDataExtractor(vector_db, FailIfInvokedLLM())

        result = extractor.extract_data_from_query(
            "ابي رسم بياني يوضح لي نسبة التحسن", k=10
        )

        self.assertEqual(result["value"].tolist(), [120.0, 55.0])
        self.assertEqual(result["currency"].tolist(), ["%", "%"])
        self.assertNotIn(3.57, result["value"].tolist())
        self.assertEqual(vector_db.queries[0][0], "نسبة التحسّن البُعد المقاس")
        self.assertEqual(vector_db.queries[0][1], len(documents))

    def test_line_chart_survives_ui_json_round_trip(self):
        chart_json = ChartGenerator.create_line_chart(
            self.df, x="label", y="value", title="Quarterly trend"
        ).to_json()

        rebuilt = go.Figure(json.loads(chart_json))

        self.assertEqual(len(rebuilt.data), 1)
        self.assertEqual(rebuilt.data[0].type, "scatter")
        self.assertEqual(rebuilt.layout.xaxis.title.font.size, 14)
        self.assertEqual(rebuilt.layout.yaxis.title.font.size, 14)

    def test_percentage_chart_labels_axis_and_values_as_percentages(self):
        percentage_df = pd.DataFrame({
            "label": ["الوعي", "التطبيق"],
            "value": [55.0, 80.0],
            "currency": ["%", "%"],
            "suggestion": ["", ""],
        })
        self.agent.vector_db = object()
        self.agent.fast_llm = object()

        with patch.object(
            FinancialDataExtractor, "extract_data_from_query", return_value=percentage_df
        ):
            result = self.agent._build_chart("ارسم نسبة التحسن")

        figure = go.Figure(json.loads(result["chart"]))
        self.assertTrue(result["success"])
        self.assertEqual(figure.layout.yaxis.title.text, "النسبة (%)")
        self.assertEqual(figure.data[0].texttemplate, "%{y:g}%")

    def test_answer_prompt_suppresses_chart_building_suggestions_when_chart_requested(self):
        llm = CapturingTextLLM()
        engine = SelfRefiningAnswerEngine(llm, verifier_llm=llm)

        answer = engine._generate_initial(
            "ارسم لي مقارنة بين قيم الصفقات حسب المرحلة",
            "Qualified: 90000 SAR\nProspecting: 75000 SAR",
            [],
            chart_requested=True,
        )

        prompt_text = "\n".join(str(message.content) for message in llm.last_messages)
        self.assertEqual(answer, "Highest stage is Qualified.")
        self.assertIn("The system is generating a chart separately", prompt_text)
        self.assertIn("Do NOT describe how to build a chart", prompt_text)

    def test_build_chart_uses_structured_chart_type_decision(self):
        self.agent.vector_db = object()
        self.agent.fast_llm = FakeChartLLM(
            ChartDecision(chart_type="line", reason="Quarterly labels form an ordered trend")
        )
        self.agent._fast_model_limiter = None

        with patch.object(
            FinancialDataExtractor, "extract_data_from_query", return_value=self.df
        ):
            result = self.agent._build_chart("show quarterly revenue trend")

        figure = go.Figure(json.loads(result["chart"]))
        self.assertTrue(result["success"])
        self.assertEqual(figure.data[0].type, "scatter")
        self.assertEqual(figure.layout.xaxis.title.text, "Label")

    def test_arabic_month_trend_line_decision_is_allowed(self):
        monthly_df = pd.DataFrame({
            "label": ["يناير", "فبراير", "مارس"],
            "value": [100.0, 125.0, 140.0],
        })
        self.agent.vector_db = object()
        self.agent.fast_llm = FakeChartLLM(
            ChartDecision(chart_type="line", reason="Arabic query asks for monthly trend")
        )
        self.agent._fast_model_limiter = None

        with patch.object(
            FinancialDataExtractor, "extract_data_from_query", return_value=monthly_df
        ):
            result = self.agent._build_chart("الاتجاه خلال الأشهر الماضية")

        figure = go.Figure(json.loads(result["chart"]))
        self.assertTrue(result["success"])
        self.assertEqual(figure.data[0].type, "scatter")

    def test_llm_line_decision_falls_back_for_categorical_labels(self):
        categorical_df = pd.DataFrame({
            "label": ["Enterprise", "SMB", "Strategic"],
            "value": [500.0, 300.0, 450.0],
        })
        self.agent.vector_db = object()
        self.agent.fast_llm = FakeChartLLM(
            ChartDecision(chart_type="line", reason="Query says trend")
        )
        self.agent._fast_model_limiter = None

        with patch.object(
            FinancialDataExtractor, "extract_data_from_query", return_value=categorical_df
        ):
            result = self.agent._build_chart("show trend by customer segment")

        figure = go.Figure(json.loads(result["chart"]))
        self.assertTrue(result["success"])
        self.assertEqual(figure.data[0].type, "bar")

    def test_llm_pie_decision_falls_back_for_too_many_categories(self):
        many_categories_df = pd.DataFrame({
            "label": [f"Category {idx}" for idx in range(1, 16)],
            "value": [float(idx) for idx in range(1, 16)],
        })
        self.agent.vector_db = object()
        self.agent.fast_llm = FakeChartLLM(
            ChartDecision(chart_type="pie", reason="Breakdown requested")
        )
        self.agent._fast_model_limiter = None

        with patch.object(
            FinancialDataExtractor, "extract_data_from_query", return_value=many_categories_df
        ):
            result = self.agent._build_chart("breakdown of expenses by category")

        figure = go.Figure(json.loads(result["chart"]))
        self.assertTrue(result["success"])
        self.assertEqual(figure.data[0].type, "bar")

    def test_small_category_pie_decision_is_allowed(self):
        expense_df = pd.DataFrame({
            "label": ["Payroll", "Rent", "Software", "Travel"],
            "value": [60.0, 20.0, 15.0, 5.0],
        })
        self.agent.vector_db = object()
        self.agent.fast_llm = FakeChartLLM(
            ChartDecision(chart_type="pie", reason="Small category breakdown")
        )
        self.agent._fast_model_limiter = None

        with patch.object(
            FinancialDataExtractor, "extract_data_from_query", return_value=expense_df
        ):
            result = self.agent._build_chart("breakdown of expenses by category")

        figure = go.Figure(json.loads(result["chart"]))
        self.assertTrue(result["success"])
        self.assertEqual(figure.data[0].type, "pie")

    def test_pipeline_stage_comparison_blocks_pie_even_with_few_categories(self):
        pipeline_df = pd.DataFrame({
            "label": ["Prospecting", "Proposal", "Negotiation", "Closed Won"],
            "value": [200000.0, 350000.0, 150000.0, 500000.0],
        })
        self.agent.vector_db = object()
        self.agent.fast_llm = FakeChartLLM(
            ChartDecision(chart_type="pie", reason="Small set of stages")
        )
        self.agent._fast_model_limiter = None

        with patch.object(
            FinancialDataExtractor, "extract_data_from_query", return_value=pipeline_df
        ):
            result = self.agent._build_chart("compare deal values across pipeline stages")

        figure = go.Figure(json.loads(result["chart"]))
        self.assertTrue(result["success"])
        self.assertEqual(figure.data[0].type, "bar")

    def test_arabic_pipeline_stage_comparison_blocks_pie(self):
        pipeline_df = pd.DataFrame({
            "label": ["تأهيل", "عرض", "تفاوض", "مغلقة"],
            "value": [200000.0, 350000.0, 150000.0, 500000.0],
        })
        self.agent.vector_db = object()
        self.agent.fast_llm = FakeChartLLM(
            ChartDecision(chart_type="pie", reason="Small set of stages")
        )
        self.agent._fast_model_limiter = None

        with patch.object(
            FinancialDataExtractor, "extract_data_from_query", return_value=pipeline_df
        ):
            result = self.agent._build_chart("قارن قيم الصفقات حسب المرحلة")

        figure = go.Figure(json.loads(result["chart"]))
        self.assertTrue(result["success"])
        self.assertEqual(figure.data[0].type, "bar")

    def test_exact_arabic_draw_stage_values_query_renders_bar(self):
        pipeline_df = pd.DataFrame({
            "label": ["Prospecting", "Qualified", "Negotiation", "Closed"],
            "value": [75000.0, 90000.0, 63000.0, 20000.0],
        })
        self.agent.vector_db = object()
        self.agent.fast_llm = FakeChartLLM(
            ChartDecision(chart_type="bar", reason="Deal values by discrete stage")
        )
        self.agent._fast_model_limiter = None

        with patch.object(
            FinancialDataExtractor, "extract_data_from_query", return_value=pipeline_df
        ):
            result = self.agent._build_chart("ارسم لي مقارنة بين قيم الصفقات حسب المرحلة")

        figure = go.Figure(json.loads(result["chart"]))
        self.assertTrue(result["success"])
        self.assertEqual(figure.data[0].type, "bar")
        self.assertEqual(list(figure.data[0].x), ["Prospecting", "Qualified", "Negotiation", "Closed"])
        self.assertEqual(
            [row["value"] for row in result["data_preview"]],
            [75000.0, 90000.0, 63000.0, 20000.0],
        )

    def test_budget_actuals_verified_calculations_are_computed_by_code(self):
        documents = [
            Document(
                page_content=(
                    "CSV: test_2_budget_actuals.xlsx\n"
                    "line_item: SaaS Revenue | type: revenue | budget_sar: 1200000 | actual_sar: 980000"
                ),
                metadata={"source": "test_2_budget_actuals.xlsx", "source_file": "test_2_budget_actuals.xlsx"},
            ),
            Document(
                page_content=(
                    "CSV: test_2_budget_actuals.xlsx\n"
                    "line_item: Services Revenue | type: revenue | budget_sar: 400000 | actual_sar: 455000"
                ),
                metadata={"source": "test_2_budget_actuals.xlsx", "source_file": "test_2_budget_actuals.xlsx"},
            ),
            Document(
                page_content=(
                    "CSV: test_2_budget_actuals.xlsx\n"
                    "line_item: Cloud Infrastructure | type: expense | budget_sar: 150000 | actual_sar: 210000"
                ),
                metadata={"source": "test_2_budget_actuals.xlsx", "source_file": "test_2_budget_actuals.xlsx"},
            ),
            Document(
                page_content=(
                    "CSV: test_2_budget_actuals.xlsx\n"
                    "line_item: Marketing | type: expense | budget_sar: 300000 | actual_sar: 180000"
                ),
                metadata={"source": "test_2_budget_actuals.xlsx", "source_file": "test_2_budget_actuals.xlsx"},
            ),
            Document(
                page_content=(
                    "CSV: test_2_budget_actuals.xlsx\n"
                    "line_item: Salaries | type: expense | budget_sar: 900000 | actual_sar: 905000"
                ),
                metadata={"source": "test_2_budget_actuals.xlsx", "source_file": "test_2_budget_actuals.xlsx"},
            ),
            Document(
                page_content=(
                    "CSV: test_2_budget_actuals.xlsx\n"
                    "line_item: Travel | type: expense | budget_sar: 50000 | actual_sar: 71000"
                ),
                metadata={"source": "test_2_budget_actuals.xlsx", "source_file": "test_2_budget_actuals.xlsx"},
            ),
            Document(
                page_content=(
                    "CSV: test_2_budget_actuals.xlsx\n"
                    "line_item: Office Rent | type: expense | budget_sar: 90000 | actual_sar: 90000"
                ),
                metadata={"source": "test_2_budget_actuals.xlsx", "source_file": "test_2_budget_actuals.xlsx"},
            ),
            Document(
                page_content=(
                    "CSV: test_2_budget_actuals.xlsx\n"
                    "line_item: Software Licenses | type: expense | budget_sar: 60000 | actual_sar: 58000"
                ),
                metadata={"source": "test_2_budget_actuals.xlsx", "source_file": "test_2_budget_actuals.xlsx"},
            ),
        ]

        block = DeterministicContextAnalyzer(documents).build_verified_context_block(
            "What is the FP&A budget actual variance and net profit?",
            AggregationIntent(needs_aggregation=True, aggregation_type="net"),
        )

        self.assertIn("Total budgeted expenses (test_2_budget_actuals.xlsx): 1,550,000 SAR", block)
        self.assertIn("Total actual expenses (test_2_budget_actuals.xlsx): 1,514,000 SAR", block)
        self.assertIn("Budgeted net profit (test_2_budget_actuals.xlsx): +50,000 SAR", block)
        self.assertIn("Actual net profit (test_2_budget_actuals.xlsx): -79,000 SAR", block)

    def test_duplicate_invoice_anomaly_is_mandatory_context(self):
        documents = [
            Document(
                page_content=(
                    "CSV: test_3_revops_data.csv\n"
                    "record_type: invoice | id: INV-501 | customer: Acme Retail | "
                    "amount_sar: 12000 | status: paid"
                ),
                metadata={"source": "test_3_revops_data.csv", "source_file": "test_3_revops_data.csv"},
            ),
            Document(
                page_content=(
                    "CSV: test_3_revops_data.csv\n"
                    "record_type: invoice | id: INV-504 | customer: Acme Retail | "
                    "amount_sar: 12000 | status: paid"
                ),
                metadata={"source": "test_3_revops_data.csv", "source_file": "test_3_revops_data.csv"},
            ),
        ]

        block = DeterministicContextAnalyzer(documents).build_verified_context_block(
            "Analyze revenue collection and billing leaks",
            AggregationIntent(needs_aggregation=True, aggregation_type="sum"),
        )

        self.assertIn("VERIFIED_ANOMALIES", block)
        self.assertIn("Potential duplicate invoice in test_3_revops_data.csv: INV-501, INV-504", block)
        self.assertIn("verify before counting both as separate revenue", block)

    def test_bank_statement_total_is_computed_by_code(self):
        amounts = [5000, 7300, 8000, 6000, 4000, 7000]
        documents = [
            Document(
                page_content=f"CSV: test_1_bank_statement.csv\npayment_id: B-{idx} | amount_sar: {amount}",
                metadata={"source": "test_1_bank_statement.csv", "source_file": "test_1_bank_statement.csv"},
            )
            for idx, amount in enumerate(amounts, start=1)
        ]

        block = DeterministicContextAnalyzer(documents).build_verified_context_block(
            "What is the bank statement total?",
            AggregationIntent(needs_aggregation=True, aggregation_type="sum"),
        )

        self.assertIn("Bank statement total (test_1_bank_statement.csv): 37,300 SAR", block)

    def test_source_name_matching_allows_human_bank_statement_claim(self):
        self.assertFalse(
            SelfRAGVerifier._source_names_conflict(
                "bank statement",
                "test_1_bank_statement.csv",
            )
        )
        self.assertTrue(
            SelfRAGVerifier._source_names_conflict(
                "bank statement",
                "test_1_ledger.csv",
            )
        )


if __name__ == "__main__":
    unittest.main()
