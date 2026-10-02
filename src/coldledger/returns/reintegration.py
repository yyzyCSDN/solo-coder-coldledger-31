"""Return reintegration gating.

A returned lot may only go back to saleable stock when temperature evidence
covers the *whole* external custody window [shipped, returned] as one
continuous interval. Two individually acceptable telemetry segments with an
uncovered stretch between them do not qualify: the uncovered stretch is an
external-custody gap that nobody monitored.

The custody handoff timeline, telemetry coverage and the return lifecycle
state are therefore evaluated against the same continuous window. When a gap
is found the only choices are quarantine or supplemental evidence; stock may
never be changed before the disposition record is closed (see
``ReintegrationBoard``).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from ..custody.handoffs import Handoff, validate as validate_handoff_chain

ENTRY_STATE = "assessed"
RELEASE = "release"
QUARANTINE = "quarantine"
SUPPLEMENT_EVIDENCE = "supplement_evidence"


@dataclass(frozen=True)
class Evidence:
    start: datetime
    end: datetime
    complete: bool
    score: float


@dataclass(frozen=True)
class Case:
    shipped: datetime
    returned: datetime
    sealed: bool
    evidence: tuple[Evidence, ...] = ()
    origin: str | None = None
    handoffs: tuple[Handoff, ...] = ()
    state: str = ENTRY_STATE


@dataclass(frozen=True)
class CoverageGap:
    """An uncovered sub-interval of the custody window."""
    start: datetime
    end: datetime
    holder: str | None = None


@dataclass(frozen=True)
class ReintegrationDecision:
    allowed: bool
    disposition: str
    reasons: tuple[str, ...]
    gaps: tuple[CoverageGap, ...]


@dataclass(frozen=True)
class ReintegrationRecord:
    reference: str
    disposition: str
    at: datetime
    decision: ReintegrationDecision | None = None


def _utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


def _as_tuple(value: Any) -> tuple:
    if isinstance(value, tuple):
        return value
    if value is None:
        return ()
    if isinstance(value, (Evidence, Handoff)):
        return (value,)
    return tuple(value)


def _holder_at(
    segments: list[tuple[datetime, datetime, str | None]], at: datetime
) -> str | None:
    for start, end, party in segments:
        if start <= at < end:
            return party
    return None


def assess(case: Case, max_score: float) -> ReintegrationDecision:
    """Evaluate reintegration against one continuous custody window."""
    start, end = _utc(case.shipped), _utc(case.returned)
    if end <= start:
        return ReintegrationDecision(
            False, QUARANTINE, ("custody window is empty or reversed",), ()
        )

    reasons: list[str] = []
    if not case.sealed:
        reasons.append("returned seal is not intact")
    if case.state != ENTRY_STATE:
        reasons.append(f"case state {case.state!r} is not {ENTRY_STATE!r}")

    # Custody handoff timeline: one unbroken chain fully inside the window.
    # Segments partition [shipped, returned] by holder so any telemetry gap
    # can be attributed to the party that had custody at the time.
    segments: list[tuple[datetime, datetime, str | None]] = []
    handoffs = _as_tuple(case.handoffs)
    if handoffs:
        chain_ok = True
        try:
            validate_handoff_chain(case.origin, case.handoffs)
        except ValueError as exc:
            reasons.append(f"custody chain broken: {exc}")
            chain_ok = False
        if chain_ok:
            party = case.origin
            seg_start = start
            for handoff in sorted(case.handoffs, key=lambda h: h.sequence):
                at = _utc(handoff.at)
                if at < start or at > end:
                    reasons.append("custody handoff falls outside return window")
                    chain_ok = False
                    break
                segments.append((seg_start, at, party))
                party, seg_start = handoff.to_party, at
            if chain_ok:
                segments.append((seg_start, end, party))
    else:
        segments.append((start, end, case.origin))

    # Telemetry coverage: clip conforming records to the window and merge.
    merged: list[list[datetime]] = []
    total_score = 0.0
    for evidence in _as_tuple(case.evidence):
        seg_start, seg_end = _utc(evidence.start), _utc(evidence.end)
        if seg_end <= seg_start:
            reasons.append("telemetry record has a non-positive interval")
            continue
        if not evidence.complete:
            reasons.append("telemetry record is incomplete")
            continue
        clipped_start, clipped_end = max(seg_start, start), min(seg_end, end)
        if clipped_end <= clipped_start:
            continue
        total_score += evidence.score
        if merged and clipped_start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], clipped_end)
        else:
            merged.append([clipped_start, clipped_end])

    # Remaining uncovered intervals of the one continuous window.
    uncovered: list[tuple[datetime, datetime]] = []
    if not merged:
        uncovered.append((start, end))
    else:
        if merged[0][0] > start:
            uncovered.append((start, merged[0][0]))
        cursor = merged[0][1]
        for seg_start, seg_end in merged[1:]:
            if seg_start > cursor:
                uncovered.append((cursor, seg_start))
            cursor = max(cursor, seg_end)
        if cursor < end:
            uncovered.append((cursor, end))

    gaps = tuple(
        CoverageGap(
            gap_start,
            gap_end,
            _holder_at(segments, gap_start) if segments else None,
        )
        for gap_start, gap_end in uncovered
    )

    if reasons:
        # Seals, chain, malformed records or lifecycle state cannot be fixed
        # by adding more telemetry, so the lot must be isolated.
        return ReintegrationDecision(False, QUARANTINE, tuple(reasons), gaps)

    if gaps:
        # Coverage is the only problem: quarantine, or supplement the missing
        # telemetry and re-assess. Stock stays untouched until that resolves.
        return ReintegrationDecision(
            False,
            SUPPLEMENT_EVIDENCE,
            ("telemetry does not continuously cover the custody window",),
            gaps,
        )

    if total_score > max_score:
        return ReintegrationDecision(
            False,
            QUARANTINE,
            (f"telemetry score {total_score} exceeds limit {max_score}",),
            (),
        )

    return ReintegrationDecision(
        True, RELEASE, ("continuous custody coverage within temperature limits",), ()
    )


def may_reintegrate(case: Case, max_score: float) -> bool:
    return assess(case, max_score).allowed


class ReintegrationBoard:
    """Orders reintegration workflow so records always precede stock moves."""

    def __init__(self) -> None:
        self._records: dict[str, ReintegrationRecord] = {}

    def record(self, reference: str) -> ReintegrationRecord | None:
        return self._records.get(reference)

    def quarantine(
        self, reference: str, decision: ReintegrationDecision, at: datetime
    ) -> ReintegrationRecord:
        if decision.disposition == RELEASE:
            raise ValueError("a releasable case must not be quarantined")
        if reference in self._records:
            raise ValueError("duplicate reintegration reference")
        record = ReintegrationRecord(reference, QUARANTINE, _utc(at), decision)
        self._records[reference] = record
        return record

    def supplement(
        self, reference: str, decision: ReintegrationDecision, at: datetime
    ) -> ReintegrationRecord:
        if decision.disposition != SUPPLEMENT_EVIDENCE:
            raise ValueError("supplemental evidence is only valid for a coverage gap")
        if reference in self._records:
            raise ValueError("duplicate reintegration reference")
        record = ReintegrationRecord(reference, SUPPLEMENT_EVIDENCE, _utc(at), decision)
        self._records[reference] = record
        return record

    def release(
        self,
        reference: str,
        case: Case,
        max_score: float,
        at: datetime,
        restock: Callable[[str], Any] | None = None,
    ) -> ReintegrationRecord:
        decision = assess(case, max_score)
        if decision.disposition != RELEASE:
            raise ValueError(
                f"cannot reintegrate: {decision.disposition}; "
                f"quarantine or supplemental evidence required"
            )
        prior = self._records.get(reference)
        if prior is not None and prior.disposition != SUPPLEMENT_EVIDENCE:
            raise ValueError("reference already has a closed disposition")
        # The disposition record is committed first; only then may stock move.
        record = ReintegrationRecord(reference, RELEASE, _utc(at), decision)
        self._records[reference] = record
        if restock is not None:
            self.restock(reference, restock)
        return record

    def restock(self, reference: str, apply: Callable[[str], Any]) -> Any:
        record = self._records.get(reference)
        if record is None:
            raise RuntimeError(
                "refusing to change stock before a disposition record exists"
            )
        if record.disposition != RELEASE:
            raise RuntimeError(
                f"reference is {record.disposition}, not released for reintegration"
            )
        return apply(reference)
