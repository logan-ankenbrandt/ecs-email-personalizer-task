"""Cost telemetry helpers.

Translates token counts into USD using the per-model rates in config.py.
Aggregates per-phase (writer/judge/refine/research) and per-run.
"""

import logging
from collections import defaultdict
from threading import Lock
from typing import Dict

from config import COST_PER_M_INPUT, COST_PER_M_OUTPUT

logger = logging.getLogger(__name__)


def usd_for_tokens(model: str, input_tokens: int, output_tokens: int) -> float:
    """Convert token counts to USD using config.py rates."""
    in_rate = COST_PER_M_INPUT.get(model, 5.0)  # conservative default
    out_rate = COST_PER_M_OUTPUT.get(model, 25.0)
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


class CostAccumulator:
    """Thread-safe accumulator for per-phase token counts and dollar costs.

    Used by the personalizer pipeline to track running totals across
    parallel per-recipient work units and report a final summary.
    """

    def __init__(self):
        self._lock = Lock()
        self._tokens: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"input": 0, "output": 0}
        )

    def record(self, phase: str, model: str, input_tokens: int, output_tokens: int) -> None:
        """Add token counts for a (phase, model) pair."""
        key = f"{phase}:{model}"
        with self._lock:
            self._tokens[key]["input"] += input_tokens
            self._tokens[key]["output"] += output_tokens

    def usd_total(self) -> float:
        """Sum all (phase, model) costs in USD."""
        total = 0.0
        with self._lock:
            for key, counts in self._tokens.items():
                model = key.split(":", 1)[1]
                total += usd_for_tokens(model, counts["input"], counts["output"])
        return total

    def usd_so_far_running(self) -> float:
        """Same as usd_total — convenience for status endpoints."""
        return self.usd_total()

    def summary(self) -> Dict[str, dict]:
        """Detailed per-phase breakdown for final logging + Mongo metadata."""
        result: Dict[str, dict] = {}
        with self._lock:
            for key, counts in self._tokens.items():
                phase, model = key.split(":", 1)
                cost = usd_for_tokens(model, counts["input"], counts["output"])
                result[key] = {
                    "phase": phase,
                    "model": model,
                    "input_tokens": counts["input"],
                    "output_tokens": counts["output"],
                    "cost_usd": round(cost, 4),
                }
        result["_total_usd"] = round(self.usd_total(), 4)
        return result
