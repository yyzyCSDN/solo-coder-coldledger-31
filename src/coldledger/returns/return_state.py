from __future__ import annotations
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Iterable

STATES = ['requested', 'authorized', 'received', 'assessed', 'quarantined',
          'reintegrated', 'closed']

# Attribute keys anchoring the assessment record to the state timeline.
ASSESSMENT_VERDICT = 'assessment.verdict'
ASSESSMENT_FINGERPRINT = 'assessment.fingerprint'
ASSESSMENT_REASONS = 'assessment.reasons'
ASSESSMENT_GAP_SECONDS = 'assessment.gap_seconds'
QUARANTINE_REASON = 'quarantine.reason'

# Explicit transition map. reintegrated may only be reached from a
# completed assessment (assessed), never directly from receipt: that was
# the path that let stock move before the custody/coverage record existed.
# quarantined stock either gets supplemental evidence and is re-assessed
# (quarantined -> assessed), or is destroyed/closed.
ALLOWED = {
    'requested': frozenset({'authorized'}),
    'authorized': frozenset({'received', 'closed'}),
    'received': frozenset({'assessed', 'quarantined'}),
    'assessed': frozenset({'reintegrated', 'quarantined', 'closed'}),
    'quarantined': frozenset({'assessed', 'closed'}),
    'reintegrated': frozenset({'closed'}),
    'closed': frozenset(),
}


@dataclass(frozen=True)
class ReturnState:
    reference: str
    state: str
    revision: int = 0
    owner: str | None = None
    updated_at: datetime | None = None
    attributes: tuple[tuple[str, str], ...] = ()

    def validate(self) -> None:
        if self.state not in STATES:
            raise ValueError("unknown state")
        if self.revision < 0:
            raise ValueError("negative revision")

@dataclass(frozen=True)
class Command:
    reference: str
    expected_revision: int
    target: str
    actor: str
    at: datetime
    reason: str = ""

@dataclass(frozen=True)
class Transition:
    reference: str
    before: str
    after: str
    revision: int
    actor: str
    at: datetime
    reason: str


def _attrs(state: ReturnState) -> dict[str, str]:
    return dict(state.attributes)


def apply(current: ReturnState, command: Command) -> tuple[ReturnState, Transition]:
    current.validate()
    if command.reference != current.reference:
        raise ValueError("reference mismatch")
    if command.expected_revision != current.revision:
        raise RuntimeError("optimistic concurrency conflict")
    if command.target not in STATES:
        raise ValueError("invalid target state")
    if command.target == current.state:
        raise ValueError("no-op transition")
    allowed = ALLOWED.get(current.state, frozenset())
    if command.target not in allowed:
        raise ValueError(f"transition {current.state} -> {command.target} not allowed")

    # Gate the inventory-impacting transitions on a recorded verdict so the
    # ledger can never be updated ahead of the assessment record.
    data = _attrs(current)
    if command.target == 'reintegrated':
        if data.get(ASSESSMENT_VERDICT) != 'release':
            raise ValueError("reintegration requires a recorded release verdict")
        if not data.get(ASSESSMENT_FINGERPRINT):
            raise ValueError("release verdict missing evidence fingerprint")
    if command.target == 'quarantined' and not data.get(QUARANTINE_REASON):
        raise ValueError("quarantine requires a recorded reason")

    at = command.at.astimezone(timezone.utc)
    previous_at = current.updated_at.astimezone(timezone.utc) if current.updated_at else None
    if previous_at is not None and at < previous_at:
        raise ValueError("event time moved backward")
    nxt = replace(current, state=command.target, revision=current.revision + 1, owner=command.actor, updated_at=at)
    event = Transition(current.reference, current.state, command.target, nxt.revision, command.actor, at, command.reason)
    return nxt, event


def replay(initial: ReturnState, events: Iterable[Transition]) -> ReturnState:
    state = initial
    expected = initial.revision + 1
    for event in sorted(events, key=lambda e: e.at):
        if event.reference != state.reference:
            raise ValueError("mixed references")
        if event.revision != expected:
            raise ValueError("revision gap")
        if event.before != state.state:
            raise ValueError("event before-state mismatch")
        if event.after not in ALLOWED.get(state.state, frozenset()):
            raise ValueError("event contains illegal transition")
        state = replace(state, state=event.after, revision=event.revision, owner=event.actor, updated_at=event.at)
        expected += 1
    return state


def can_transition(current: ReturnState, target: str) -> bool:
    return target in ALLOWED.get(current.state, frozenset())


def next_states(current: ReturnState) -> tuple[str, ...]:
    return tuple(sorted(ALLOWED.get(current.state, frozenset())))


def state_age_seconds(current: ReturnState, now: datetime) -> float | None:
    if current.updated_at is None:
        return None
    value = (now.astimezone(timezone.utc) - current.updated_at.astimezone(timezone.utc)).total_seconds()
    return max(0.0, value)


def with_attribute(current: ReturnState, key: str, value: str) -> ReturnState:
    data = dict(current.attributes)
    data[key] = value
    return replace(current, attributes=tuple(sorted(data.items())))


def attribute(current: ReturnState, key: str, default: str | None = None) -> str | None:
    return dict(current.attributes).get(key, default)
