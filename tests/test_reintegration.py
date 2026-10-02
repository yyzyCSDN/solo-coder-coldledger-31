import unittest
from datetime import datetime, timezone, timedelta

from coldledger.returns.reintegration import (
    Evidence, Case, Handoff, assess, may_reintegrate,
    CoverageGap, ReintegrationBoard,
    RELEASE, QUARANTINE, SUPPLEMENT_EVIDENCE,
)

UTC = timezone.utc
T0 = datetime(2026, 10, 1, 8, tzinfo=UTC)


def ev(start_h, end_h, score=0.0, complete=True):
    return Evidence(T0 + timedelta(hours=start_h), T0 + timedelta(hours=end_h),
                    complete, score)


class ContinuousCoverageTests(unittest.TestCase):
    def test_two_segments_with_uncovered_middle_is_rejected(self):
        # Regression: both segments are individually acceptable, but the
        # external custody stretch between them has no telemetry at all.
        case = Case(
            shipped=T0, returned=T0 + timedelta(hours=10), sealed=True,
            evidence=(ev(0, 4), ev(6, 10)),
        )
        decision = assess(case, 5.0)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.disposition, SUPPLEMENT_EVIDENCE)
        self.assertEqual(len(decision.gaps), 1)
        self.assertEqual(decision.gaps[0].start, T0 + timedelta(hours=4))
        self.assertEqual(decision.gaps[0].end, T0 + timedelta(hours=6))
        self.assertFalse(may_reintegrate(case, 5.0))

    def test_abutting_segments_cover_continuously(self):
        case = Case(
            shipped=T0, returned=T0 + timedelta(hours=10), sealed=True,
            evidence=(ev(0, 5), ev(5, 10, score=1.0)),
        )
        decision = assess(case, 5.0)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.disposition, RELEASE)
        self.assertEqual(decision.gaps, ())

    def test_overlapping_segments_cover_continuously(self):
        case = Case(
            shipped=T0, returned=T0 + timedelta(hours=10), sealed=True,
            evidence=(ev(0, 7), ev(4, 10)),
        )
        self.assertTrue(may_reintegrate(case, 5.0))

    def test_evidence_short_of_window_edges(self):
        case = Case(
            shipped=T0, returned=T0 + timedelta(hours=10), sealed=True,
            evidence=(ev(1, 9)),
        )
        decision = assess(case, 5.0)
        self.assertFalse(decision.allowed)
        self.assertEqual(len(decision.gaps), 2)

    def test_score_limit_still_enforced(self):
        case = Case(
            shipped=T0, returned=T0 + timedelta(hours=10), sealed=True,
            evidence=(ev(0, 10, score=9.0)),
        )
        decision = assess(case, 5.0)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.disposition, QUARANTINE)

    def test_incomplete_evidence_quarantines(self):
        case = Case(
            shipped=T0, returned=T0 + timedelta(hours=10), sealed=True,
            evidence=(ev(0, 10, complete=False)),
        )
        self.assertEqual(assess(case, 5.0).disposition, QUARANTINE)

    def test_broken_seal_quarantines_even_when_covered(self):
        case = Case(
            shipped=T0, returned=T0 + timedelta(hours=10), sealed=False,
            evidence=(ev(0, 10)),
        )
        self.assertEqual(assess(case, 5.0).disposition, QUARANTINE)

    def test_wrong_state_quarantines(self):
        case = Case(
            shipped=T0, returned=T0 + timedelta(hours=10), sealed=True,
            evidence=(ev(0, 10)), state="received",
        )
        self.assertEqual(assess(case, 5.0).disposition, QUARANTINE)

    def test_gap_attributed_to_custody_holder_from_handoff_timeline(self):
        handoffs = (
            Handoff(1, "plant", "carrier", T0 + timedelta(hours=2), "S1"),
            Handoff(2, "carrier", "depot", T0 + timedelta(hours=5), "S2"),
            Handoff(3, "depot", "carrier", T0 + timedelta(hours=7), "S3"),
            Handoff(4, "carrier", "plant", T0 + timedelta(hours=10), "S4"),
        )
        # Gap from hour 3 to hour 6: plant/carrier handover at 2, depot takes
        # custody at 5 -> the uncovered stretch belongs to the carrier.
        case = Case(
            shipped=T0, returned=T0 + timedelta(hours=10), sealed=True,
            evidence=(ev(0, 3), ev(6, 10)),
            origin="plant", handoffs=handoffs,
        )
        decision = assess(case, 5.0)
        self.assertEqual(len(decision.gaps), 1)
        self.assertEqual(decision.gaps[0].holder, "carrier")

    def test_broken_custody_chain_quarantines(self):
        handoffs = (
            Handoff(1, "carrier", "depot", T0 + timedelta(hours=5), "S1"),
        )
        case = Case(
            shipped=T0, returned=T0 + timedelta(hours=10), sealed=True,
            evidence=(ev(0, 10)), origin="plant", handoffs=handoffs,
        )
        self.assertEqual(assess(case, 5.0).disposition, QUARANTINE)

    def test_handoff_outside_window_quarantines(self):
        handoffs = (
            Handoff(1, "plant", "carrier", T0 - timedelta(hours=1), "S1"),
            Handoff(2, "carrier", "plant", T0 + timedelta(hours=10), "S2"),
        )
        case = Case(
            shipped=T0, returned=T0 + timedelta(hours=10), sealed=True,
            evidence=(ev(0, 10)), origin="plant", handoffs=handoffs,
        )
        self.assertEqual(assess(case, 5.0).disposition, QUARANTINE)


class BoardWorkflowTests(unittest.TestCase):
    def _gap_case(self):
        return Case(
            shipped=T0, returned=T0 + timedelta(hours=10), sealed=True,
            evidence=(ev(0, 4), ev(6, 10)),
        )

    def test_release_applies_stock_after_record(self):
        board = ReintegrationBoard()
        case = Case(shipped=T0, returned=T0 + timedelta(hours=10),
                    sealed=True, evidence=(ev(0, 10),))
        moves = []
        board.release("R1", case, 5.0, T0 + timedelta(hours=11),
                      restock=lambda ref: moves.append(ref))
        self.assertEqual(moves, ["R1"])
        self.assertEqual(board.record("R1").disposition, RELEASE)

    def test_stock_change_refused_before_record(self):
        board = ReintegrationBoard()
        with self.assertRaises(RuntimeError):
            board.restock("R2", lambda ref: None)

    def test_stock_change_refused_after_quarantine(self):
        board = ReintegrationBoard()
        decision = assess(self._gap_case(), 5.0)
        board.quarantine("R3", decision, T0 + timedelta(hours=11))
        with self.assertRaises(RuntimeError):
            board.restock("R3", lambda ref: None)

    def test_gap_allows_supplement_then_release(self):
        board = ReintegrationBoard()
        decision = assess(self._gap_case(), 5.0)
        board.supplement("R4", decision, T0 + timedelta(hours=11))
        # Cannot restock while evidence is still outstanding.
        with self.assertRaises(RuntimeError):
            board.restock("R4", lambda ref: None)
        closed = Case(
            shipped=T0, returned=T0 + timedelta(hours=10), sealed=True,
            evidence=(ev(0, 4), ev(4, 6), ev(6, 10)),
        )
        moves = []
        board.release("R4", closed, 5.0, T0 + timedelta(days=1),
                      restock=lambda ref: moves.append(ref))
        self.assertEqual(moves, ["R4"])

    def test_direct_release_blocked_when_gap_exists(self):
        board = ReintegrationBoard()
        with self.assertRaises(ValueError):
            board.release("R5", self._gap_case(), 5.0,
                          T0 + timedelta(hours=11))
        self.assertIsNone(board.record("R5"))


if __name__ == "__main__":
    unittest.main()
