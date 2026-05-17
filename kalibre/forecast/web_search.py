"""Phase 6B: tiny env-driven config + evidence-hash helper.

Pure dataclass. The runner reads :func:`WebSearchConfig.from_env` once
per pass and pipes the resolved knobs into ``build_default_provider`` /
``build_opus_provider``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kalibre.budget import BudgetProfile


DEFAULT_ENGINE = "exa"
DEFAULT_MAX_RESULTS = 4
DEFAULT_MAX_TOTAL_RESULTS = 12
DEFAULT_CONTEXT_SIZE = "low"
DEFAULT_DEADLINE_BUFFER_SEC = 210.0
DEFAULT_DAILY_HARD_CAP_USD = 10.0
DEFAULT_SEARCH_UNIT_COST_USD = 0.005


@dataclass(frozen=True)
class WebSearchConfig:
    """Phase 6B env-driven knobs for OpenRouter native web search.

    All defaults reproduce the pre-6B (search-off) behaviour. When
    ``enabled`` is False the runner does not pass ``tools`` to the
    provider; everything remains identical to Phase 6R2.
    """

    enabled: bool = False
    engine: str = DEFAULT_ENGINE
    max_results: int = DEFAULT_MAX_RESULTS
    max_total_results: int = DEFAULT_MAX_TOTAL_RESULTS
    context_size: str = DEFAULT_CONTEXT_SIZE
    deadline_buffer_sec: float = DEFAULT_DEADLINE_BUFFER_SEC
    daily_hard_cap_usd: float = DEFAULT_DAILY_HARD_CAP_USD
    unit_cost_usd: float = DEFAULT_SEARCH_UNIT_COST_USD
    excluded_domains: tuple[str, ...] = ("reddit.com",)
    allowed_domains: tuple[str, ...] = ()
    # Phase 6C: when ``min_universe_for_search`` markets or fewer survive
    # the universe filter, treat the tick as too-thin-to-search-on and
    # fall back to a non-search forecast (or skip entirely). 0 disables
    # the short-circuit. Default 0 preserves Phase 6B behaviour; the
    # operator opts in via env. Use 1 for "skip when universe gave us
    # 1 or fewer markets".
    min_universe_for_search: int = 0

    @classmethod
    def from_env(
        cls,
        env: dict[str, str] | None = None,
        *,
        budget_profile: "BudgetProfile | None" = None,
    ) -> "WebSearchConfig":
        env_map = env if env is not None else dict(os.environ)
        # Phase 6D: the budget profile's web-search defaults apply only
        # when the per-knob env var is unset. Operators can still pin a
        # value via env to override profile defaults.
        prof = budget_profile
        prof_max_results = (
            prof.web_search_max_results if prof is not None else DEFAULT_MAX_RESULTS
        )
        prof_max_total = (
            prof.web_search_max_total_results if prof is not None else DEFAULT_MAX_TOTAL_RESULTS
        )
        prof_daily_cap = (
            prof.web_search_daily_hard_cap_usd if prof is not None else DEFAULT_DAILY_HARD_CAP_USD
        )
        enabled = (env_map.get("KALIBRE_WEB_SEARCH_MODE", "").strip() == "1")
        engine = (env_map.get("KALIBRE_WEB_SEARCH_ENGINE") or DEFAULT_ENGINE).strip() or DEFAULT_ENGINE
        try:
            max_results = int(env_map.get("KALIBRE_WEB_SEARCH_MAX_RESULTS") or prof_max_results)
        except ValueError:
            max_results = prof_max_results
        try:
            max_total = int(env_map.get("KALIBRE_WEB_SEARCH_MAX_TOTAL_RESULTS") or prof_max_total)
        except ValueError:
            max_total = prof_max_total
        context = (env_map.get("KALIBRE_WEB_SEARCH_CONTEXT_SIZE") or DEFAULT_CONTEXT_SIZE).strip().lower() or DEFAULT_CONTEXT_SIZE
        try:
            buffer_sec = float(env_map.get("KALIBRE_WEB_SEARCH_DEADLINE_BUFFER_SEC") or DEFAULT_DEADLINE_BUFFER_SEC)
        except ValueError:
            buffer_sec = DEFAULT_DEADLINE_BUFFER_SEC
        try:
            daily_cap = float(env_map.get("KALIBRE_WEB_SEARCH_DAILY_HARD_CAP_USD") or prof_daily_cap)
        except ValueError:
            daily_cap = prof_daily_cap
        try:
            unit_cost = float(env_map.get("KALIBRE_WEB_SEARCH_UNIT_COST_USD") or DEFAULT_SEARCH_UNIT_COST_USD)
        except ValueError:
            unit_cost = DEFAULT_SEARCH_UNIT_COST_USD
        try:
            min_universe = int(
                env_map.get("KALIBRE_WEB_SEARCH_MIN_UNIVERSE") or 0,
            )
        except ValueError:
            min_universe = 0
        excluded_raw = env_map.get("KALIBRE_WEB_SEARCH_EXCLUDED_DOMAINS")
        if excluded_raw is None:
            excluded = ("reddit.com",)
        else:
            excluded = tuple(
                d.strip() for d in excluded_raw.split(",") if d.strip()
            )
        allowed_raw = env_map.get("KALIBRE_WEB_SEARCH_ALLOWED_DOMAINS")
        allowed = (
            tuple(d.strip() for d in allowed_raw.split(",") if d.strip())
            if allowed_raw else ()
        )
        return cls(
            enabled=enabled,
            engine=engine,
            max_results=max(1, max_results),
            max_total_results=max(1, max_total),
            context_size=context,
            deadline_buffer_sec=max(0.0, buffer_sec),
            daily_hard_cap_usd=max(0.0, daily_cap),
            unit_cost_usd=max(0.0, unit_cost),
            excluded_domains=excluded,
            allowed_domains=allowed,
            min_universe_for_search=max(0, min_universe),
        )

    def to_provider_overrides(self) -> dict[str, Any]:
        """Flat dict consumed by ``build_default_provider`` /
        ``build_opus_provider`` to construct a ProviderConfig with the
        Phase 6B fields set."""
        return {
            "web_search_enabled": self.enabled,
            "web_search_engine": self.engine,
            "web_search_max_results": self.max_results,
            "web_search_max_total_results": self.max_total_results,
            "web_search_context_size": self.context_size,
            "web_search_allowed_domains": self.allowed_domains,
            "web_search_excluded_domains": self.excluded_domains,
            "web_search_unit_cost_usd": self.unit_cost_usd,
        }


def compute_evidence_hash(
    cited_urls: tuple[str, ...] | list[str],
    web_search_requests: int,
) -> str:
    """Stable hash of the evidence that fed this forecast. Used by the
    forecast cache key so two evidence-bearing forecasts with different
    cited URLs don't collide."""
    urls_sorted = sorted({u.strip() for u in (cited_urls or ()) if u})
    payload = "|".join(urls_sorted) + f"#req={int(web_search_requests or 0)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
