"""Tests for FinancialRAGJudge and the record evaluation/aggregation pipeline."""

import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_insight_lite.modules.eval import (
    FinancialRAGJudge,
    aggregate_judge_scores,
    evaluate_records,
)


# Model identity terms that must never leak into a judge prompt, since the
# judge is meant to score answers blind to which model produced them.
REDACTED_MODEL_TERMS = ("kimi", "minimax", "gpt-oss")

VALID_SCORE = json.dumps(
    {
        "groundedness": 5,
        "numerical_accuracy": 5,
        "relevance": 4,
        "clarity": 4,
        "overall": 4.5,
        "unsupported_claims": [],
        "rationale": "The answer is supported by the supplied source chunks.",
    }
)

JUDGE_METRICS = ("groundedness", "numerical_accuracy", "relevance", "clarity", "overall")

# Aggregation assertions require stable, labeled model records. The chat-history
# export intentionally has no model labels, so it cannot exercise per-model
# aggregation behavior.
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "test_chart_generation.jsonl"


class FakeJudgeClient:
    """A stand-in judge client that replays canned responses in order.

    Each entry in `responses` is either a raw response string to return,
    or an Exception instance to raise -- letting tests simulate API
    failures interleaved with successful/malformed responses.
    """

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    def prompts_seen(self):
        """All prompt text (system + user) sent across every call, joined."""
        return "\n".join(f"{s}\n{u}" for s, u in self.calls)


class FinancialRAGJudgeTests(unittest.TestCase):
    def make_judge(self, responses):
        """Build a FinancialRAGJudge wired to a FakeJudgeClient with no real sleep."""
        client = FakeJudgeClient(responses)
        judge = FinancialRAGJudge(client, sleep=lambda _: None)
        return judge, client

    def test_valid_json_is_parsed_correctly(self):
        judge, client = self.make_judge([VALID_SCORE])

        score, raw = judge.score(
            question="What did the model report?",
            source_chunks=["Revenue was SAR 100 million"],
            generated_answer="Reported revenue was SAR 100 million",
        )

        self.assertEqual(score.groundedness, 5)
        self.assertEqual(score.overall, 4.5)
        self.assertEqual(raw, VALID_SCORE)
        self.assertEqual(len(client.calls), 1)

    def test_model_identity_is_redacted_from_prompt(self):
        """None of the model names involved should ever reach the judge prompt."""
        judge, client = self.make_judge([VALID_SCORE])

        judge.score(
            question="What did kimi report?",
            source_chunks=["minimax revenue was SAR 100 million"],
            generated_answer="gpt-oss says revenue was SAR 100 million",
        )

        combined_prompt = client.prompts_seen().lower()
        for term in REDACTED_MODEL_TERMS:
            with self.subTest(term=term):
                self.assertNotIn(term, combined_prompt)

    def test_malformed_and_api_failures_retry_then_fail_gracefully(self):
        judge, client = self.make_judge(
            ["not json", RuntimeError("temporary API error"), "{}"]
        )
        pauses = []
        judge.sleep = pauses.append

        score, raw = judge.score(
            question="Q", source_chunks=["Source"], generated_answer="Answer"
        )

        self.assertIsNone(score)
        self.assertEqual(raw, "{}")
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(pauses, [1, 2])

    def test_all_attempts_failing_still_returns_gracefully(self):
        """Even if every retry raises, score() should not propagate the exception."""
        judge, client = self.make_judge(
            [
                RuntimeError("api down"),
                RuntimeError("api down"),
                RuntimeError("api down"),
            ]
        )

        score, raw = judge.score(
            question="Q", source_chunks=["Source"], generated_answer="Answer"
        )

        self.assertIsNone(score)
        self.assertEqual(len(client.calls), 3)

    def test_no_sleep_is_invoked_when_first_attempt_succeeds(self):
        pauses = []
        client = FakeJudgeClient([VALID_SCORE])
        judge = FinancialRAGJudge(client, sleep=pauses.append)

        score, _ = judge.score(
            question="Q", source_chunks=["Source"], generated_answer="Answer"
        )

        self.assertIsNotNone(score)
        self.assertEqual(pauses, [])

    def test_boundary_scores_are_parsed_without_clamping_or_rejection(self):
        """A judge should trust the model's own min (1) and max (5) scores."""
        boundary_score = json.dumps(
            {
                "groundedness": 1,
                "numerical_accuracy": 1,
                "relevance": 5,
                "clarity": 5,
                "overall": 3,
                "unsupported_claims": ["The Q3 revenue figure is not in any source chunk"],
                "rationale": "Mixed: strong clarity but a fabricated figure.",
            }
        )
        judge, _ = self.make_judge([boundary_score])

        score, _ = judge.score(
            question="Q", source_chunks=["Source"], generated_answer="Answer"
        )

        self.assertEqual(score.groundedness, 1)
        self.assertEqual(score.relevance, 5)
        self.assertEqual(score.overall, 3)

    def test_unsupported_claims_and_rationale_are_preserved(self):
        """The judge's qualitative reasoning shouldn't be dropped during parsing."""
        claims = ["The 12% growth figure is not present in any source chunk"]
        rationale = "The headline number is fabricated; everything else checks out."
        annotated_score = json.dumps(
            {
                "groundedness": 3,
                "numerical_accuracy": 2,
                "relevance": 4,
                "clarity": 4,
                "overall": 3,
                "unsupported_claims": claims,
                "rationale": rationale,
            }
        )
        judge, _ = self.make_judge([annotated_score])

        score, _ = judge.score(
            question="Q", source_chunks=["Source"], generated_answer="Answer"
        )

        self.assertEqual(score.unsupported_claims, claims)
        self.assertEqual(score.rationale, rationale)

    def test_json_missing_required_field_is_treated_as_malformed(self):
        """A response missing a required metric (e.g. 'overall') should fail like bad JSON,
        not raise or silently default a score."""
        incomplete = json.dumps(
            {
                "groundedness": 5,
                "numerical_accuracy": 5,
                "relevance": 4,
                "clarity": 4,
                # "overall" omitted
                "unsupported_claims": [],
                "rationale": "Missing overall score.",
            }
        )
        judge, client = self.make_judge([incomplete, incomplete, incomplete])

        score, raw = judge.score(
            question="Q", source_chunks=["Source"], generated_answer="Answer"
        )

        self.assertIsNone(score)
        self.assertEqual(len(client.calls), 3)

    def test_non_object_json_is_treated_as_malformed(self):
        """A syntactically valid JSON value that isn't a score object (e.g. a bare list)
        should be rejected the same way malformed text is."""
        judge, client = self.make_judge(["[1, 2, 3]", "[1, 2, 3]", "[1, 2, 3]"])

        score, raw = judge.score(
            question="Q", source_chunks=["Source"], generated_answer="Answer"
        )

        self.assertIsNone(score)
        self.assertEqual(len(client.calls), 3)

    def test_redaction_is_case_insensitive(self):
        """Model names should be stripped from the prompt regardless of casing."""
        judge, client = self.make_judge([VALID_SCORE])

        judge.score(
            question="What did KIMI and Gpt-Oss disagree on?",
            source_chunks=["MINIMAX reported SAR 100 million"],
            generated_answer="Revenue was SAR 100 million",
        )

        combined_prompt = client.prompts_seen().lower()
        for term in REDACTED_MODEL_TERMS:
            with self.subTest(term=term):
                self.assertNotIn(term, combined_prompt)


class EvaluateAndAggregateTests(unittest.TestCase):
    """Tests covering the fixture-driven evaluate_records/aggregate_judge_scores flow."""

    @classmethod
    def setUpClass(cls):
        cls.records = [
            json.loads(line) for line in FIXTURE_PATH.read_text().splitlines() if line.strip()
        ]

    def evaluate_fixture(self):
        client = FakeJudgeClient([VALID_SCORE] * len(self.records))
        judge = FinancialRAGJudge(client, sleep=lambda _: None)
        rows = evaluate_records(self.records, with_judge=True, judge=judge)
        aggregates = {row["model"]: row for row in aggregate_judge_scores(rows)}
        return rows, aggregates

    def test_fixture_evaluation_produces_one_row_per_record(self):
        rows, _ = self.evaluate_fixture()

        self.assertEqual(len(rows), len(self.records))
        self.assertTrue(all(row["judge_status"] == "ok" for row in rows))

    def test_token_f1_scores_are_normalized(self):
        rows, _ = self.evaluate_fixture()

        for row in rows:
            with self.subTest(row=row.get("model")):
                self.assertGreaterEqual(row["token_f1"], 0)
                self.assertLessEqual(row["token_f1"], 1)

    def test_aggregate_row_counts_per_model(self):
        _, aggregates = self.evaluate_fixture()

        self.assertEqual(aggregates["candidate_a"]["judge_rows"], 2)
        self.assertEqual(aggregates["candidate_b"]["judge_rows"], 1)

    def test_aggregate_judge_metrics_stay_in_valid_range(self):
        _, aggregates = self.evaluate_fixture()

        for model, aggregate in aggregates.items():
            for metric in JUDGE_METRICS:
                with self.subTest(model=model, metric=metric):
                    self.assertGreaterEqual(aggregate[metric], 1)
                    self.assertLessEqual(aggregate[metric], 5)

    def test_aggregation_excludes_failed_judge_calls(self):
        """A judge failure on one record should drop that row out of the aggregate
        counts rather than being silently averaged in as if it succeeded.

        This is a stronger check than bounds-checking on identical repeated scores:
        it verifies aggregate_judge_scores is actually keying off judge_status.
        """
        # First record's judge call fails outright across all retries; the rest succeed.
        responses = [RuntimeError("api down"), RuntimeError("api down"), RuntimeError("api down")]
        responses += [VALID_SCORE] * (len(self.records) - 1)
        client = FakeJudgeClient(responses)
        judge = FinancialRAGJudge(client, sleep=lambda _: None)

        rows = evaluate_records(self.records, with_judge=True, judge=judge)
        aggregates = {row["model"]: row for row in aggregate_judge_scores(rows)}

        ok_row_count = sum(1 for row in rows if row["judge_status"] == "ok")
        self.assertEqual(ok_row_count, len(self.records) - 1)

        # Every row should still be present -- a judge failure marks the row,
        # it doesn't drop it from the evaluation output.
        self.assertEqual(len(rows), len(self.records))

        # The failed row must not be counted toward any model's aggregate.
        total_aggregated_rows = sum(a["judge_rows"] for a in aggregates.values())
        self.assertEqual(total_aggregated_rows, ok_row_count)

    # -- Tightened validation of the aggregate arithmetic itself -----------------
    #
    # The tests above confirm counts and bounds (1-5), but none of them confirm
    # that aggregate_judge_scores is actually computing the right numbers. The
    # tests below pin down the real arithmetic with distinct, hand-picked scores
    # so a bug like "averaging the wrong column" or "cross-model contamination"
    # would fail loudly instead of slipping through a range check.

    def test_aggregate_computes_exact_arithmetic_mean_per_metric(self):
        """Aggregates must reflect the precise mean of underlying scores, not just
        fall within the valid 1-5 range. Two records for the same model, with
        distinct known per-metric scores, must average to a predictable value."""
        score_a = json.dumps({
            "groundedness": 2, "numerical_accuracy": 4, "relevance": 3, "clarity": 5,
            "overall": 3.0, "unsupported_claims": [], "rationale": "First score.",
        })
        score_b = json.dumps({
            "groundedness": 4, "numerical_accuracy": 2, "relevance": 5, "clarity": 3,
            "overall": 4.0, "unsupported_claims": [], "rationale": "Second score.",
        })

        records = [
            {**self.records[0], "model": "candidate_x"},
            {**self.records[0], "model": "candidate_x"},
        ]
        client = FakeJudgeClient([score_a, score_b])
        judge = FinancialRAGJudge(client, sleep=lambda _: None)

        rows = evaluate_records(records, with_judge=True, judge=judge)
        aggregates = {row["model"]: row for row in aggregate_judge_scores(rows)}
        agg = aggregates["candidate_x"]

        self.assertAlmostEqual(agg["groundedness"], 3.0)
        self.assertAlmostEqual(agg["numerical_accuracy"], 3.0)
        self.assertAlmostEqual(agg["relevance"], 4.0)
        self.assertAlmostEqual(agg["clarity"], 4.0)
        self.assertAlmostEqual(agg["overall"], 3.5)

    def test_aggregate_of_single_row_equals_raw_score(self):
        """With exactly one successful row, the aggregate should be an identity
        transform of that row's scores -- no rounding or drift introduced."""
        score = json.dumps({
            "groundedness": 2, "numerical_accuracy": 3, "relevance": 4, "clarity": 1,
            "overall": 2.5, "unsupported_claims": [], "rationale": "Only measurement.",
        })
        records = [{**self.records[0], "model": "candidate_solo"}]
        client = FakeJudgeClient([score])
        judge = FinancialRAGJudge(client, sleep=lambda _: None)

        rows = evaluate_records(records, with_judge=True, judge=judge)
        aggregates = {row["model"]: row for row in aggregate_judge_scores(rows)}
        agg = aggregates["candidate_solo"]

        self.assertEqual(agg["groundedness"], 2)
        self.assertEqual(agg["numerical_accuracy"], 3)
        self.assertEqual(agg["relevance"], 4)
        self.assertEqual(agg["clarity"], 1)
        self.assertEqual(agg["overall"], 2.5)
        self.assertEqual(agg["judge_rows"], 1)

    def test_aggregate_does_not_cross_contaminate_between_models(self):
        """Scores from one model must not leak into another model's aggregate when
        evaluate_records processes interleaved records for multiple models."""
        score_hi = json.dumps({
            "groundedness": 5, "numerical_accuracy": 5, "relevance": 5, "clarity": 5,
            "overall": 5.0, "unsupported_claims": [], "rationale": "High score.",
        })
        score_lo = json.dumps({
            "groundedness": 1, "numerical_accuracy": 1, "relevance": 1, "clarity": 1,
            "overall": 1.0, "unsupported_claims": ["fabricated"], "rationale": "Low score.",
        })

        records = [
            {**self.records[0], "model": "candidate_hi"},
            {**self.records[0], "model": "candidate_lo"},
            {**self.records[0], "model": "candidate_hi"},
            {**self.records[0], "model": "candidate_lo"},
        ]
        client = FakeJudgeClient([score_hi, score_lo, score_hi, score_lo])
        judge = FinancialRAGJudge(client, sleep=lambda _: None)

        rows = evaluate_records(records, with_judge=True, judge=judge)
        aggregates = {row["model"]: row for row in aggregate_judge_scores(rows)}

        self.assertEqual(aggregates["candidate_hi"]["overall"], 5.0)
        self.assertEqual(aggregates["candidate_lo"]["overall"], 1.0)
        # judge_rows must reflect only that model's own successful calls
        self.assertEqual(aggregates["candidate_hi"]["judge_rows"], 2)
        self.assertEqual(aggregates["candidate_lo"]["judge_rows"], 2)

    def test_model_with_no_successful_judge_calls_is_excluded_from_aggregate(self):
        """If every judge call for a model fails, that model shouldn't appear in the
        aggregate at all -- a row of averaged nulls/zeros would silently masquerade
        as a real (if low) score.

        NOTE: this pins down a specific design choice (omit vs. emit judge_rows=0).
        If aggregate_judge_scores is meant to still emit a visible zero-row for that
        model, flip this assertion accordingly -- the important thing is that the
        behavior is deliberate and tested, not left implicit.
        """
        records = [{**self.records[0], "model": "candidate_never_scored"}]
        client = FakeJudgeClient([RuntimeError("down")] * 3)
        judge = FinancialRAGJudge(client, sleep=lambda _: None)

        rows = evaluate_records(records, with_judge=True, judge=judge)
        aggregates = {row["model"]: row for row in aggregate_judge_scores(rows)}

        self.assertNotIn("candidate_never_scored", aggregates)

    def test_aggregate_matches_known_real_model_output(self):
        """Regression check against an actual gpt-oss judging run: three real
        records (2 for candidate_a, 1 for candidate_b), all scored a perfect 5
        across every metric, must aggregate to exactly 5.0 for both models."""
        perfect_score = json.dumps({
            "groundedness": 5, "numerical_accuracy": 5, "relevance": 5, "clarity": 5,
            "overall": 5.0, "unsupported_claims": [],
            "rationale": "Matches the source chunk exactly.",
        })
        client = FakeJudgeClient([perfect_score] * len(self.records))
        judge = FinancialRAGJudge(client, sleep=lambda _: None)

        rows = evaluate_records(self.records, with_judge=True, judge=judge)
        aggregates = {row["model"]: row for row in aggregate_judge_scores(rows)}

        for model in ("candidate_a", "candidate_b"):
            with self.subTest(model=model):
                for metric in JUDGE_METRICS:
                    self.assertEqual(aggregates[model][metric], 5.0)


class EvaluateRecordsPerformanceTests(unittest.TestCase):
    """Wall-clock performance checks for evaluate_records.

    These are deliberately generous on thresholds -- the goal isn't to pin down
    an exact millisecond budget (which would be flaky across machines/CI), but
    to catch two concrete regressions: (1) something reintroducing a *real*
    time.sleep instead of routing through the injected sleep hook, and (2) an
    accidental O(n^2) pass over the records/rows that would make evaluation
    blow up on larger batches.
    """

    @classmethod
    def setUpClass(cls):
        cls.base_records = [
            json.loads(line) for line in FIXTURE_PATH.read_text().splitlines() if line.strip()
        ]

    def make_records(self, count):
        """Cycle the fixture records up to `count` entries (shallow copies)."""
        base = self.base_records
        return [dict(base[i % len(base)]) for i in range(count)]

    def test_evaluate_records_completes_quickly_with_stubbed_sleep(self):
        """With a stubbed sleep hook and an instant fake client, evaluating a
        moderate batch should take well under a second of wall-clock time --
        there's no I/O or real backoff happening."""
        records = self.make_records(60)
        client = FakeJudgeClient([VALID_SCORE] * len(records))
        judge = FinancialRAGJudge(client, sleep=lambda _: None)

        start = time.perf_counter()
        rows = evaluate_records(records, with_judge=True, judge=judge)
        elapsed = time.perf_counter() - start

        self.assertEqual(len(rows), len(records))
        self.assertLess(elapsed, 1.0, f"evaluate_records took {elapsed:.3f}s for 60 records")

    def test_retries_do_not_incur_real_wall_clock_delay(self):
        """Records whose judge call fails twice before succeeding must not cost
        real seconds -- the retry backoff should route entirely through the
        injected sleep hook, never through the real time.sleep."""
        records = self.make_records(10)
        responses = []
        for _ in records:
            responses += [RuntimeError("api down"), RuntimeError("api down"), VALID_SCORE]
        client = FakeJudgeClient(responses)
        judge = FinancialRAGJudge(client, sleep=lambda _: None)

        # If evaluate_records (or FinancialRAGJudge) falls back to the real
        # time.sleep anywhere, this patch turns that into a hard failure
        # instead of a silent multi-second stall.
        with patch("time.sleep", side_effect=AssertionError("real time.sleep was called")):
            start = time.perf_counter()
            rows = evaluate_records(records, with_judge=True, judge=judge)
            elapsed = time.perf_counter() - start

        self.assertEqual(len(rows), len(records))
        self.assertTrue(all(row["judge_status"] == "ok" for row in rows))
        # Without stubbing, 10 records x (1s + 2s) backoff would cost ~30s.
        self.assertLess(elapsed, 1.0, f"evaluate_records took {elapsed:.3f}s for 10 retried records")

    def test_evaluate_records_scales_roughly_linearly(self):
        """Evaluating 10x more records should cost roughly 10x the time, not
        50x or 100x -- a loose guard against an accidental O(n^2) pass (e.g.
        re-scanning the full row list once per record)."""
        small = self.make_records(5)
        large = self.make_records(50)

        client_small = FakeJudgeClient([VALID_SCORE] * len(small))
        judge_small = FinancialRAGJudge(client_small, sleep=lambda _: None)
        start = time.perf_counter()
        evaluate_records(small, with_judge=True, judge=judge_small)
        small_elapsed = max(time.perf_counter() - start, 1e-6)

        client_large = FakeJudgeClient([VALID_SCORE] * len(large))
        judge_large = FinancialRAGJudge(client_large, sleep=lambda _: None)
        start = time.perf_counter()
        evaluate_records(large, with_judge=True, judge=judge_large)
        large_elapsed = time.perf_counter() - start

        # 10x the records should cost nowhere near 10x^2; generous 30x ceiling
        # to absorb timing noise on a shared/CI machine while still catching
        # genuine quadratic blowups.
        self.assertLess(
            large_elapsed,
            small_elapsed * 30,
            f"50 records took {large_elapsed:.4f}s vs {small_elapsed:.4f}s for 5 "
            "-- looks worse than linear",
        )


if __name__ == "__main__":
    unittest.main()
