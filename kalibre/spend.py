"""T0 spend governor.

The cheap kill-switch that makes T0 safe: no LLM/model call escapes
without a positive yes from :meth:`SpendGovernor.can_call`. Returns the
reason when blocked so the loop can log it.

T0 doesn't actually make model calls yet, but having the governor wired
in and tested now means later tiers only have to call it.

Cap hierarchy in :meth:`can_call` (first match wins; all blocked
decisions return a reason starting with ``"no model call allowed"``):

1. Negative cost -> always blocked.
2. **Total actual hard cap** (``spent_total + est > total_budget_usd``)
   -> blocks every model tier.
3. **Daily hard cap** (``spent_day + est > daily_hard_usd``) -> blocks
   every model tier.
4. **Projected total hard cap** (``projected_total + est >
   total_budget_usd``) -> blocks every model tier. Only meaningful when
   ``eval_end_ts`` is set; otherwise the projection collapses to
   ``spent_total`` and step 2 fires first.
5. Daily soft cap with opus -> blocks ``opus`` only.
6. Projected total > 95% of budget -> blocks ``opus`` only (degradation
   hint).

The 70/85/95% degradation thresholds in :meth:`degraded_mode` are a
*separate* concept driven by actual ``spent_total`` only, not by
projection. They surface a recommended degradation mode but do not
themselves block calls.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class SpendDecision:
    allowed: bool
    reason: str

    def __bool__(self) -> bool:
        return self.allowed


class SpendGovernor:
    """Caps total + per-day spend and degrades by model tier as budget burns."""

    BLOCK_REASON_PREFIX = "no model call allowed"

    def __init__(
        self,
        *,
        daily_soft_usd: float = 25.0,
        daily_hard_usd: float = 35.0,
        total_budget_usd: float = 500.0,
        eval_end_ts: datetime | None = None,
    ) -> None:
        if daily_soft_usd < 0 or daily_hard_usd < 0 or total_budget_usd < 0:
            raise ValueError("caps must be non-negative")
        if daily_soft_usd > daily_hard_usd:
            raise ValueError("daily soft cap cannot exceed hard cap")
        self.daily_soft_usd = float(daily_soft_usd)
        self.daily_hard_usd = float(daily_hard_usd)
        self.total_budget_usd = float(total_budget_usd)
        self.eval_end_ts = eval_end_ts
        self._spent_total = 0.0
        self._spent_day = 0.0
        self._day_key: str | None = None
        self._lock = threading.Lock()

    # --- public api ---------------------------------------------------------

    def can_call(
        self,
        model: str | None,
        est_cost_usd: float,
        *,
        now: datetime | None = None,
    ) -> SpendDecision:
        with self._lock:
            if est_cost_usd < 0:
                return SpendDecision(
                    False, f"{self.BLOCK_REASON_PREFIX}: negative cost",
                )
            now = now or datetime.now(tz=UTC)
            self._rotate_day_locked(now)
            tier = (model or "").lower() or "structural_only"

            # 2. Total actual hard cap -> blocks every tier.
            if self._spent_total + est_cost_usd > self.total_budget_usd:
                return SpendDecision(
                    False, f"{self.BLOCK_REASON_PREFIX}: total budget exhausted",
                )

            # 3. Daily hard cap -> blocks every tier.
            if self._spent_day + est_cost_usd > self.daily_hard_usd:
                return SpendDecision(
                    False, f"{self.BLOCK_REASON_PREFIX}: daily hard cap exceeded",
                )

            # 4. Projected total hard cap -> blocks every tier.
            projected = self._project_total_locked(now)
            if projected + est_cost_usd > self.total_budget_usd:
                return SpendDecision(
                    False,
                    f"{self.BLOCK_REASON_PREFIX}: projected total over hard budget",
                )

            # 5. Daily soft cap -> blocks opus only.
            if (
                self._spent_day + est_cost_usd > self.daily_soft_usd
                and tier == "opus"
            ):
                return SpendDecision(
                    False,
                    f"{self.BLOCK_REASON_PREFIX}: daily soft cap (opus blocked)",
                )

            # 6. Projected > 95% of total -> blocks opus only.
            if (
                projected + est_cost_usd > self.total_budget_usd * 0.95
                and tier == "opus"
            ):
                return SpendDecision(
                    False,
                    f"{self.BLOCK_REASON_PREFIX}: projected total above 95% (opus blocked)",
                )

            return SpendDecision(True, "allowed")

    def record(self, cost_usd: float, *, now: datetime | None = None) -> None:
        if cost_usd < 0:
            raise ValueError("cost_usd must be non-negative")
        with self._lock:
            now = now or datetime.now(tz=UTC)
            self._rotate_day_locked(now)
            self._spent_total += float(cost_usd)
            self._spent_day += float(cost_usd)

    def degraded_mode(self) -> str:
        with self._lock:
            if self._spent_total > self.total_budget_usd * 0.95:
                return "structural_only"
            if self._spent_total > self.total_budget_usd * 0.85:
                return "sonnet_only"
            if self._spent_total > self.total_budget_usd * 0.70:
                return "sonnet_plus_haiku"
            return "full"

    @property
    def spent_total_usd(self) -> float:
        with self._lock:
            return self._spent_total

    @property
    def spent_day_usd(self) -> float:
        with self._lock:
            return self._spent_day

    # --- internals ----------------------------------------------------------

    def _rotate_day_locked(self, now: datetime) -> None:
        key = now.astimezone(UTC).date().isoformat()
        if self._day_key != key:
            self._day_key = key
            self._spent_day = 0.0

    def _project_total_locked(self, now: datetime) -> float:
        if self.eval_end_ts is None:
            return self._spent_total
        remaining_days = max(0, (self.eval_end_ts - now).days)
        return self._spent_total + self._spent_day * remaining_days
