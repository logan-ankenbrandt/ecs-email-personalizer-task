"""Budget tracking for orchestrator sessions.

Three independent caps, checked together via has_room():
  - max_turns: top-level orchestrator tool-use turns (sub-agents have
    their own internal turn budgets; those don't count here).
  - max_usd: dollar spend across orchestrator + all sub-agents.
  - max_sec: wall-clock ceiling.

Pattern ported from Claude Code's maxTurns + maxBudgetUsd + taskBudget
trio (QueryEngine.ts:148). The three ceilings are deliberately not
redundant: `max_turns` catches pathological orchestrator indecision,
`max_usd` catches expensive sub-agent escalation, `max_sec` catches web
research stalls.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Budget:
    """Per-recipient orchestrator budget.

    Call tick_turn() after every API turn and spend(usd) after every LLM
    call (orchestrator or sub-agent). Call has_room() before each turn.
    When has_room() returns False, the orchestrator should gracefully
    finish any pending submit_step/skip_step actions and end the session.
    """

    max_turns: int = 12
    # Round 4 Tier E.2: raised from 0.60 to 0.90. Post-Tier-B compaction,
    # target is still $0.45-$0.60 but this gives the orchestrator a genuine
    # soft landing before the hard stop, instead of making the target also
    # be the ceiling.
    max_usd: float = 0.90
    max_sec: int = 600

    turn_count: int = 0
    cost_usd: float = 0.0
    started_at: float = field(default_factory=time.monotonic)

    def tick_turn(self) -> None:
        self.turn_count += 1

    def spend(self, usd: float) -> None:
        self.cost_usd += usd

    def elapsed_sec(self) -> float:
        return time.monotonic() - self.started_at

    def has_room(self) -> bool:
        return (
            self.turn_count < self.max_turns
            and self.cost_usd < self.max_usd
            and self.elapsed_sec() < self.max_sec
        )

    def reason_exhausted(self) -> Optional[str]:
        """Return the specific ceiling that was hit, or None if budget
        still has room. Used for log lines and skip reasons."""
        if self.turn_count >= self.max_turns:
            return "max_turns"
        if self.cost_usd >= self.max_usd:
            return "max_usd"
        if self.elapsed_sec() >= self.max_sec:
            return "max_sec"
        return None

    def summary(self) -> dict:
        """One-line telemetry dump for end-of-session logging."""
        return {
            "turns": self.turn_count,
            "cost_usd": round(self.cost_usd, 4),
            "elapsed_sec": round(self.elapsed_sec(), 1),
            "exhausted": self.reason_exhausted(),
        }
