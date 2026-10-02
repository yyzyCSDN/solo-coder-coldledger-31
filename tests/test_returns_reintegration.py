import unittest
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from coldledger.custody.handoffs import Handoff
from coldledger.inventory.ledger import InventoryLedger
from coldledger.returns import service
from coldledger.returns.reintegration import (
    Assessment, Case, Evidence, assess, fingerprint, may_reintegrate,
    RELEASE, QUARANTINE,
)
from coldledger.returns.return_state import (
    STATES, ALLOWED, ASSESSMENT_VERDICT, ASSESSMENT_FINGERPRINT, QUARANTINE_REASON,
    Command, ReturnState, apply, attribute,
)

UTC = timezone.utc
T0 = datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
T1 = T0 + timedelta(hours=12)


def ev(s_h, e_h, score=1.0, complete=True, source=""):
    return Evidence(T0 + timedelta(hours=s_h), T0 + timedelta(hours=e_h),
                    complete, score, source)


def case_with_gap(handoffs=()):
    # Two individually compliant segments; hours 4..8 are uncovered.
    return Case(
        T0, T1, sealed=True,
        evidence=(ev(0, 4, source="carrier-A"), ev(8, 12, source="carrier-B")),
        handoffs=handoffs,
    )


def continuous_case():
    return Case(
        T0, T1, sealed=True,
        evidence=(ev(0, 6, source="carrier-A"), ev(6, 12, source="carrier-B")),
    )


class CoverageTests(unittest.TestCase):
    def test_two_good_segments_with_gap_do_not_pass(self):
        result = assess(case_with_gap(), max_score=5.0)
        self.assertEqual(result.verdict, QUARANTINE)
        self.assertEqual(result.gap_seconds, 4 * 3600)
        self.assertEqual(len(result.gaps), 1)
        g = result.gaps[0]
        self.assertEqual(g.start, T0 + timedelta(hours=4))
        self.assertEqual(g.end, T0 + timedelta(hours=8))
        self.assertFalse(may_reintegrate(case_with_gap(), 5.0))

    def test_old_boolean_gate_regression(self):
        # The previous cursor-based logic returned True here even with
        # four uncovered hours between the two segments.
        self.assertFalse(may_reintegrate(case_with_gap(), max_score=100.0))

    def test_continuous_coverage_releases(self):
        result = assess(continuous_case(), max_score=5.0)
        self.assertEqual(result.verdict, RELEASE)
        self.assertEqual(result.gap_seconds, 0.0)
        self.assertEqual(result.gaps, ())
        self.assertTrue(may_reintegrate(continuous_case(), 5.0))

    def test_every_segment_must_individually_comply(self):
        c = Case(T0, T1, True, (ev(0, 12, score=9.0),))
        self.assertFalse(may_reintegrate(c, max_score=5.0))
        result = assess(c, max_score=5.0)
        self.assertEqual(result.worst_score, 9.0)
        self.assertTrue(any("exceeds limit" in r for r in result.reasons))

    def test_incomplete_segment_covers_nothing(self):
        c = Case(T0, T1, True, (ev(0, 12, complete=False),))
        result = assess(c, max_score=5.0)
        self.assertEqual(result.verdict, QUARANTINE)
        self.assertEqual(result.gap_seconds, 12 * 3600)

    def test_broken_seal_quarantines_even_with_full_coverage(self):
        c = Case(T0, T1, sealed=False, evidence=(ev(0, 12),))
        self.assertFalse(may_reintegrate(c, 5.0))

    def test_overlapping_segments_merge_no_double_count(self):
        c = Case(T0, T1, True, (ev(0, 7), ev(5, 12)))
        result = assess(c, max_score=5.0)
        self.assertEqual(result.verdict, RELEASE)
        self.assertEqual(result.covered_seconds, 12 * 3600)

    def test_evidence_outside_window_rejected(self):
        c = Case(T0, T1, True, (ev(0, 4), ev(7, 13)))
        result = assess(c, max_score=5.0)
        self.assertEqual(result.verdict, QUARANTINE)
        self.assertTrue(any("outside custody window" in r for r in result.reasons))

    def test_gap_attributed_to_custodian_at_that_time(self):
        handoffs = (
            Handoff(1, "depot", "carrier-A", T0, "seal-1"),
            Handoff(2, "carrier-A", "depot", T0 + timedelta(hours=6), "seal-2"),
        )
        result = assess(case_with_gap(handoffs), max_score=5.0)
        self.assertEqual(result.verdict, QUARANTINE)
        self.assertEqual(result.gaps[0].custodian, "depot")

    def test_handoff_outside_window_breaks_timeline(self):
        handoffs = (
            Handoff(1, "depot", "carrier-A", T0 - timedelta(hours=1), "seal-1"),
        )
        result = assess(case_with_gap(handoffs), max_score=5.0)
        self.assertEqual(result.verdict, QUARANTINE)

    def test_supplemental_evidence_closes_gap(self):
        result = assess(case_with_gap(), max_score=5.0)
        self.assertEqual(result.verdict, QUARANTINE)
        filled = service.supplement(case_with_gap(), (ev(4, 8, source="depot-recorder"),))
        result2 = assess(filled, max_score=5.0)
        self.assertEqual(result2.verdict, RELEASE)

    def test_fingerprint_changes_when_evidence_patched(self):
        c = continuous_case()
        filled = service.supplement(c, (ev(2, 3),))
        self.assertNotEqual(fingerprint(c), fingerprint(filled))


class StateMachineTests(unittest.TestCase):
    def test_states_include_quarantine(self):
        self.assertIn("quarantined", STATES)
        self.assertNotIn("reintegrated", ALLOWED["received"])
        self.assertNotIn("reintegrated", ALLOWED["quarantined"])
        self.assertIn("assessed", ALLOWED["quarantined"])
        self.assertIn("reintegrated", ALLOWED["assessed"])

    def test_cannot_reintegrate_without_recorded_release(self):
        state = ReturnState("R1", "received")
        with self.assertRaises(ValueError):
            apply(state, Command("R1", 0, "reintegrated", "qa", T1))

    def _assessed(self, ref="R1"):
        state = ReturnState(ref, "received")
        state = service.process_return(
            state, continuous_case(), max_score=5.0,
            actor="qa", at=T0 + timedelta(hours=13),
        ).state
        self.assertEqual(state.state, "assessed")
        return state

    def test_release_path_records_verdict_before_transition(self):
        state = self._assessed()
        self.assertEqual(attribute(state, ASSESSMENT_VERDICT), RELEASE)
        self.assertTrue(attribute(state, ASSESSMENT_FINGERPRINT))

    def test_gap_path_quarantines_and_never_posts_stock(self):
        ledger = InventoryLedger()
        state = ReturnState("R2", "received")
        outcome = service.process_return(
            state, case_with_gap(), max_score=5.0,
            actor="qa", at=T0 + timedelta(hours=13),
        )
        self.assertTrue(outcome.quarantined)
        self.assertEqual(outcome.state.state, "quarantined")
        self.assertTrue(attribute(outcome.state, QUARANTINE_REASON))
        self.assertEqual(ledger.entries(), ())
        with self.assertRaises(ValueError):
            service.post_reinstatement(outcome.state, ledger, "ACC-SALEABLE", 7,
                                       "qa", T0 + timedelta(hours=14))
        self.assertEqual(ledger.entries(), ())

    def test_supplement_then_release_then_ledger_in_order(self):
        ledger = InventoryLedger()
        state = ReturnState("R3", "received")
        outcome = service.process_return(
            state, case_with_gap(), 5.0, "qa", T0 + timedelta(hours=13))
        self.assertTrue(outcome.quarantined)

        filled = service.supplement(
            case_with_gap(), (ev(4, 8, source="depot-recorder"),))
        outcome2 = service.process_return(
            outcome.state, filled, 5.0, "qa", T0 + timedelta(days=1))
        self.assertTrue(outcome2.released)
        self.assertEqual(outcome2.state.state, "assessed")
        self.assertEqual(ledger.entries(), ())  # still no stock movement

        state3, transition, entry = service.post_reinstatement(
            outcome2.state, ledger, "ACC-SALEABLE", Decimal("7"),
            "qa", T0 + timedelta(days=1, hours=1), case=filled)
        self.assertEqual(state3.state, "reintegrated")
        self.assertEqual(transition.before, "assessed")
        self.assertEqual(transition.after, "reintegrated")
        self.assertEqual(entry.sequence, 1)
        self.assertEqual(ledger.balance("ACC-SALEABLE"), Decimal("7"))

    def test_post_reinstatement_is_idempotent(self):
        ledger = InventoryLedger()
        state = self._assessed("R4")
        state, _, e1 = service.post_reinstatement(
            state, ledger, "ACC", Decimal("3"), "qa", T0 + timedelta(hours=14),
            case=continuous_case())
        state, _, e2 = service.post_reinstatement(
            state, ledger, "ACC", Decimal("3"), "qa", T0 + timedelta(hours=15),
            case=continuous_case())
        self.assertIs(e1, e2)
        self.assertEqual(ledger.balance("ACC"), Decimal("3"))

    def test_patching_evidence_after_release_rejected(self):
        ledger = InventoryLedger()
        state = self._assessed("R5")
        tampered = service.supplement(continuous_case(), (ev(2, 3),))
        with self.assertRaises(ValueError):
            service.post_reinstatement(
                state, ledger, "ACC", Decimal("3"), "qa",
                T0 + timedelta(hours=14), case=tampered)
        self.assertEqual(ledger.entries(), ())

    def test_failed_reassessment_stays_quarantined_via_assessed(self):
        state = ReturnState("R6", "received")
        outcome = service.process_return(
            state, case_with_gap(), 5.0, "qa", T0 + timedelta(hours=13))
        # supplement only partially closes the gap (4..6 remains uncovered)
        partial = service.supplement(case_with_gap(), (ev(4, 6),))
        outcome2 = service.process_return(
            outcome.state, partial, 5.0, "qa", T0 + timedelta(days=1))
        self.assertEqual(outcome2.state.state, "quarantined")
        self.assertEqual(len(outcome2.transitions), 2)
        self.assertEqual(
            [(t.before, t.after) for t in outcome2.transitions],
            [("quarantined", "assessed"), ("assessed", "quarantined")],
        )
        self.assertEqual(outcome2.assessment.gap_seconds, 2 * 3600)


if __name__ == "__main__":
    unittest.main()
