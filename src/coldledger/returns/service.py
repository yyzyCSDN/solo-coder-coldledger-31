"""Return reinstatement workflow.

One continuous flow ties together the custody handoff timeline, telemetry
coverage assessment and the return state machine:

    received --(assessment)--> assessed --> reintegrated --> (inventory post)
                    \\                       ^
                     `--> quarantined ------' (only after supplemental
                           evidence re-assesses to release)

Invariant: the inventory ledger is written only after the release verdict
and its evidence fingerprint have been recorded on the state, and only by
the assessed -> reintegrated transition. Stock can therefore never be moved
"on trust" and documented later; a coverage gap forces quarantine first.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..inventory.ledger import InventoryLedger, Entry
from .reintegration import (
    RELEASE, QUARANTINE, Assessment, Case, Evidence, assess, fingerprint,
    with_evidence,
)
from .return_state import (
    ASSESSMENT_FINGERPRINT, ASSESSMENT_GAP_SECONDS, ASSESSMENT_REASONS,
    ASSESSMENT_VERDICT, QUARANTINE_REASON, Command, ReturnState, Transition,
    apply, with_attribute,
)


@dataclass(frozen=True)
class ReinstatementResult:
    state: ReturnState
    transitions: tuple[Transition, ...]
    assessment: Assessment
    entries: tuple[Entry, ...] = ()

    @property
    def quarantined(self) -> bool:
        return self.state.state == "quarantined"

    @property
    def released(self) -> bool:
        return self.state.state == "assessed"


def supplement(case: Case, extra: tuple[Evidence, ...] | list[Evidence]) -> Case:
    """Attach supplemental telemetry to a quarantined case for re-assessment."""
    return with_evidence(case, extra)


def _verdict_attributes(state: ReturnState, result: Assessment) -> ReturnState:
    state = with_attribute(state, ASSESSMENT_VERDICT, result.verdict)
    state = with_attribute(state, ASSESSMENT_FINGERPRINT, result.fingerprint)
    state = with_attribute(state, ASSESSMENT_GAP_SECONDS, str(result.gap_seconds))
    state = with_attribute(state, ASSESSMENT_REASONS, "\n".join(result.reasons))
    if result.verdict == QUARANTINE:
        reason = result.reasons[0] if result.reasons else "uncovered custody interval"
        state = with_attribute(state, QUARANTINE_REASON, reason)
    return state


def _move(state: ReturnState, target: str, actor: str, at: datetime,
          reference: str, revision: int) -> tuple[ReturnState, Transition]:
    return apply(
        state,
        Command(reference, revision, target, actor, at,
                reason=f"return assessment: {target}"),
    )


def _record_quarantine(state: ReturnState, reference: str, actor: str,
                       at: datetime) -> tuple[ReturnState, Transition]:
    if not dict(state.attributes).get(QUARANTINE_REASON):
        state = with_attribute(state, QUARANTINE_REASON, "uncovered custody interval")
    return _move(state, "quarantined", actor, at, reference, state.revision)


def process_return(state: ReturnState, case: Case, max_score: float,
                   actor: str, at: datetime,
                   tolerance_seconds: float = 0.0) -> ReinstatementResult:
    """Assess a received (or re-assess a quarantined) return.

    Records the verdict, then moves the state. Inventory is never touched
    here; a gap verdict lands in quarantine pending supplemental evidence.
    """
    if state.state not in {"received", "quarantined"}:
        raise ValueError(f"cannot assess return in state {state.state}")

    result = assess(case, max_score, at=at, tolerance_seconds=tolerance_seconds)
    reference = state.reference
    recorded = _verdict_attributes(state, result)
    transitions: list[Transition] = []

    if result.verdict == RELEASE:
        recorded, event = _move(recorded, "assessed", actor, at,
                                reference, state.revision)
        transitions.append(event)
    elif state.state == "received":
        recorded, event = _record_quarantine(recorded, reference, actor, at)
        transitions.append(event)
    else:
        # Re-assessment of quarantined stock that still fails: record the
        # fresh verdict through assessed -> quarantined so every decision
        # appears in the event timeline instead of mutating history.
        recorded, to_assessed = _move(recorded, "assessed", actor, at,
                                      reference, state.revision)
        recorded = _verdict_attributes(recorded, result)
        recorded, back_to_quarantine = _record_quarantine(
            recorded, reference, actor, at)
        transitions.extend([to_assessed, back_to_quarantine])

    return ReinstatementResult(recorded, tuple(transitions), result)


def post_reinstatement(state: ReturnState, ledger: InventoryLedger,
                       account: str, quantity, actor: str, at: datetime,
                       case: Case | None = None) -> tuple[ReturnState, Transition, Entry]:
    """Move released stock back into the ledger.

    The assessed -> reintegrated event is appended before the ledger entry.
    When ``case`` is supplied its evidence fingerprint must still match the
    assessed record: patching evidence after release is rejected, so the
    order cannot be "stock first, paperwork later".
    """
    data = dict(state.attributes)
    if state.state not in {"assessed", "reintegrated"}:
        raise ValueError(f"cannot reinstate from state {state.state}")
    if data.get(ASSESSMENT_VERDICT) != RELEASE:
        raise ValueError("no recorded release verdict for this return")
    stored_fp = data.get(ASSESSMENT_FINGERPRINT)
    if not stored_fp:
        raise ValueError("release verdict missing evidence fingerprint")
    if case is not None and fingerprint(case) != stored_fp:
        raise ValueError("evidence record changed after release assessment")

    reference = f"return:{state.reference}:reinstate"
    existing = next((e for e in ledger.entries() if e.reference == reference), None)

    transition: Transition | None = None
    if state.state == "assessed":
        state, transition = apply(
            state,
            Command(state.reference, state.revision, "reintegrated", actor, at,
                    reason="release verdict recorded; stock returned to saleable inventory"),
        )

    if existing is not None:
        return state, transition, existing
    entry = ledger.post(account, reference, quantity,
                        reason=f"return {state.reference} reinstated after continuous coverage assessment")
    return state, transition, entry


def quarantine(state: ReturnState, actor: str, at: datetime,
               reason: str) -> tuple[ReturnState, Transition]:
    """Explicit quarantine from received or assessed states."""
    if state.state not in {"received", "assessed"}:
        raise ValueError(f"cannot quarantine from state {state.state}")
    marked = with_attribute(state, QUARANTINE_REASON, reason)
    return _move(marked, "quarantined", actor, at, state.reference, state.revision)


def close(state: ReturnState, actor: str, at: datetime,
          reason: str) -> tuple[ReturnState, Transition]:
    return apply(
        state,
        Command(state.reference, state.revision, "closed", actor, at, reason=reason),
    )
