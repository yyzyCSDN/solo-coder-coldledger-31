"""Return reintegration assessment.

A returned cold-chain unit may go back to saleable stock only when the
*entire* external custody interval (shipped -> returned, across every
custody handoff) is covered by temperature evidence that is individually
within limits. Two compliant segments with an uncovered gap between them
must not pass: the gap is an unattributed external custody interval and
forces quarantine until supplemental evidence closes it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from ..core.clock import utc
from ..core.intervals import Interval, merge, subtract
from ..custody.handoffs import Handoff, validate as validate_custody

RELEASE = "release"
QUARANTINE = "quarantine"


@dataclass(frozen=True)
class Evidence:
    start: datetime
    end: datetime
    complete: bool
    score: float
    source: str = ""


@dataclass(frozen=True)
class Case:
    shipped: datetime
    returned: datetime
    sealed: bool
    evidence: tuple[Evidence, ...] = ()
    handoffs: tuple[Handoff, ...] = ()


@dataclass(frozen=True)
class CoverageGap:
    start: datetime
    end: datetime
    seconds: float
    custodian: str | None


@dataclass(frozen=True)
class Assessment:
    verdict: str
    window_start: datetime
    window_end: datetime
    covered_seconds: float
    gap_seconds: float
    worst_score: float
    gaps: tuple[CoverageGap, ...] = ()
    reasons: tuple[str, ...] = ()
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    fingerprint: str = ""

    @property
    def release(self) -> bool:
        return self.verdict == RELEASE


def _custodian_at(handoffs: tuple[Handoff, ...], moment: datetime) -> str | None:
    """Party holding the unit at ``moment`` according to the handoff timeline."""
    if not handoffs:
        return None
    current = handoffs[0].from_party
    for h in sorted(handoffs, key=lambda x: (x.at, x.sequence)):
        if h.at.astimezone(timezone.utc) <= moment:
            current = h.to_party
        else:
            break
    return current


def with_evidence(case: Case, extra: tuple[Evidence, ...] | list[Evidence]) -> Case:
    """Return a new case with supplemental evidence attached.

    Only meaningful while the return is still received/quarantined: the
    state machine refuses to move a reintegrated case backwards.
    """
    return replace(case, evidence=tuple(case.evidence) + tuple(extra))


def assess(case: Case, max_score: float, at: datetime | None = None,
           tolerance_seconds: float = 0.0) -> Assessment:
    """Assess reinstatement against one continuous custody interval."""
    if max_score < 0:
        raise ValueError("max_score must be non-negative")
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be non-negative")

    start, end = utc(case.shipped), utc(case.returned)
    window = end - start
    reasons: list[str] = []

    if not case.sealed:
        reasons.append("seal broken on return")
    if window.total_seconds() <= 0:
        reasons.append("non-positive custody window")

    handoffs = tuple(sorted(case.handoffs, key=lambda x: (x.at, x.sequence)))
    if handoffs:
        try:
            validate_custody(handoffs[0].from_party, handoffs)
            for h in handoffs:
                hat = utc(h.at)
                if hat < start or hat > end:
                    reasons.append(f"handoff {h.sequence} outside custody window")
        except ValueError as exc:
            reasons.append(f"custody chain broken: {exc}")

    # Every segment must be individually valid; an invalid segment covers
    # nothing and is never silently merged into the timeline.
    valid: list[tuple[datetime, datetime, Evidence]] = []
    if window.total_seconds() > 0:
        for e in case.evidence:
            label = e.source or "evidence"
            s, t = utc(e.start), utc(e.end)
            if t <= s:
                reasons.append(f"{label}: non-positive interval")
                continue
            if s < start or t > end:
                reasons.append(f"{label}: interval outside custody window")
                continue
            if not e.complete:
                reasons.append(f"{label}: incomplete telemetry")
                continue
            valid.append((s, t, e))
            if e.score > max_score:
                reasons.append(f"{label}: score {e.score} exceeds limit {max_score}")

    worst = max((e.score for _, _, e in valid), default=0.0)

    # Union of evidence coverage, then subtract from the custody window.
    # merge() collapses overlaps, so overlapping segments are neither
    # double-counted nor allowed to paper over a gap.
    gaps: list[CoverageGap] = []
    gap_seconds = 0.0
    if window.total_seconds() > 0:
        covered = merge([Interval(s, t) for s, t, _ in valid])
        for piece in subtract(Interval(start, end), covered):
            seconds = (piece.end - piece.start).total_seconds()
            if seconds <= tolerance_seconds:
                continue
            midpoint = piece.start + (piece.end - piece.start) / 2
            gaps.append(CoverageGap(
                piece.start, piece.end, seconds,
                _custodian_at(handoffs, midpoint),
            ))
            gap_seconds += seconds
        if gaps:
            holders = sorted({g.custodian or "unattributed" for g in gaps})
            reasons.append(
                "uncovered custody interval: "
                + ", ".join(f"{g.start.isoformat()}..{g.end.isoformat()} "
                            f"({g.custodian or 'unattributed'})" for g in gaps)
            )
            reasons.append("custodian during gap: " + ",".join(holders))

    verdict = RELEASE if not reasons else QUARANTINE
    return Assessment(
        verdict=verdict,
        window_start=start,
        window_end=end,
        covered_seconds=window.total_seconds() - gap_seconds,
        gap_seconds=gap_seconds,
        worst_score=worst,
        gaps=tuple(gaps),
        reasons=tuple(reasons),
        at=utc(at) if at else datetime.now(timezone.utc),
        fingerprint=fingerprint(case),
    )


def fingerprint(case: Case) -> str:
    """Stable hash of the exact custody/evidence record assessed."""
    lines = [
        f"shipped={utc(case.shipped).isoformat()}",
        f"returned={utc(case.returned).isoformat()}",
        f"sealed={case.sealed}",
    ]
    for e in sorted(case.evidence, key=lambda x: (utc(x.start), utc(x.end), x.source)):
        lines.append("|".join([
            "ev", utc(e.start).isoformat(), utc(e.end).isoformat(),
            str(e.complete), str(e.score), e.source,
        ]))
    for h in sorted(case.handoffs, key=lambda x: x.sequence):
        lines.append("|".join([
            "ho", str(h.sequence), h.from_party, h.to_party,
            utc(h.at).isoformat(), h.seal,
        ]))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def may_reintegrate(case: Case, max_score: float, tolerance_seconds: float = 0.0) -> bool:
    """Backward-compatible boolean gate: release only on full coverage."""
    return assess(case, max_score, tolerance_seconds=tolerance_seconds).release
