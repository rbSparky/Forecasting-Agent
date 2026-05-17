"""Per-tick forecast pass.

Glues universe selection, spend gating, deadline gating, the forecast
provider, the cache, and the calibrator. Outputs a dict of
:class:`MarketProbability` (keyed by ``market_id``) plus audit rows that
the loop persists to ``state.sqlite3``.

Behavior:

- Markets are filtered to the universe-accepted set, then truncated to
  ``max_markets``.
- Before the loop, if fewer than ``deadline_skip_buffer_sec`` seconds
  remain to the local cutoff, *all* forecast calls are skipped and an
  audit row is emitted per target market with
  ``edge_source='forecast_blocked_deadline'``.
- Within the loop, each market consults the spend governor; if blocked,
  an audit row with ``edge_source='forecast_blocked_spend'`` is emitted
  and the call is skipped.
- Successful results are calibrated via
  :func:`kalibre.calibrate.calibrate_probability`, recorded in the
  cache, and converted into :class:`MarketProbability` for the
  deterministic strategy.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable

from kalibre.budget import BudgetProfile, force_opus_enabled
from kalibre.calibrate import Calibrator, calibrate_probability
from kalibre.forecast.blend import (
    BlendResult,
    blend_with_market_prior,
    compute_market_prior,
    inverse_variance_blend,
)
from kalibre.forecast.cache import (
    DEFAULT_TTL_SEC,
    ForecastCache,
    MemoryForecastCache,
    cache_key,
)
from kalibre.forecast.config import ForecastConfig, get_default_config
from kalibre.forecast.provider import (
    ForecastProvider,
    NoOpForecastProvider,
    estimate_cost_usd,
)
from kalibre.forecast.types import ForecastBatchResult, ForecastRequest, ForecastResult
from kalibre.forecast.web_search import WebSearchConfig, compute_evidence_hash
from kalibre.layers import LayersConfig
from kalibre.selection import (
    ACCEPTED_EXPLORATION,
    SelectionPassResult,
    SelectorExplorationConfig,
    passthrough_selector,
    select_forecast_targets,
)
from kalibre.sizing import clip_sigma_p
from kalibre.spend import SpendGovernor
from kalibre.strategy import MarketProbability
from kalibre.tick_context import MarketView, TickContext
from kalibre.universe import UniverseFilterConfig, filter_universe


DEFAULT_MAX_FORECAST_MARKETS = 6
DEFAULT_DEADLINE_SKIP_BUFFER_SEC = 90.0
DEFAULT_PER_CALL_BUFFER_SEC = 30.0

# Phase 6 Opus escalation knobs.
DEFAULT_OPUS_MAX_CALLS_PER_TICK = 1
DEFAULT_OPUS_MIN_SONNET_EDGE = 0.04
DEFAULT_OPUS_DISAGREEMENT_THRESHOLD = 0.08


@dataclass(frozen=True)
class OpusEscalationConfig:
    """Phase 6 dual-model escalation.

    Sonnet remains the default forecaster. When this config is enabled,
    a small number of markets per tick are re-forecast with Opus 4
    (default cap 1 call per tick) and combined with the Sonnet result
    via inverse-variance blend. Strong disagreement collapses to a
    proposal-level ``reject_reason="model_disagreement"``.

    Read by :meth:`from_env`:

    - ``KALIBRE_OPUS_ESCALATION_MODE=0|1`` (default 0)
    - ``KALIBRE_OPUS_MAX_CALLS_PER_TICK=1`` (default 1)
    - ``KALIBRE_OPUS_FOR_EXPLORATION=1`` (default 1 when mode on)
    - ``KALIBRE_OPUS_MIN_SONNET_EDGE=0.04`` (default 0.04)
    - ``KALIBRE_OPUS_DIRECT_WHEN_TINY_CANDIDATE_SET=0|1`` (default 0)
    - ``KALIBRE_OPUS_DISAGREEMENT_THRESHOLD=0.08`` (default 0.08)
    """

    enabled: bool = False
    max_calls_per_tick: int = DEFAULT_OPUS_MAX_CALLS_PER_TICK
    for_exploration: bool = True
    min_sonnet_edge: float = DEFAULT_OPUS_MIN_SONNET_EDGE
    direct_when_tiny_candidate_set: bool = False
    disagreement_threshold: float = DEFAULT_OPUS_DISAGREEMENT_THRESHOLD

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "OpusEscalationConfig":
        env_map = env if env is not None else dict(os.environ)
        enabled = (env_map.get("KALIBRE_OPUS_ESCALATION_MODE", "").strip() == "1")
        try:
            max_calls = int(
                env_map.get("KALIBRE_OPUS_MAX_CALLS_PER_TICK")
                or DEFAULT_OPUS_MAX_CALLS_PER_TICK,
            )
        except ValueError:
            max_calls = DEFAULT_OPUS_MAX_CALLS_PER_TICK
        for_exploration = (
            env_map.get("KALIBRE_OPUS_FOR_EXPLORATION", "1").strip() != "0"
        )
        try:
            min_edge = float(
                env_map.get("KALIBRE_OPUS_MIN_SONNET_EDGE")
                or DEFAULT_OPUS_MIN_SONNET_EDGE,
            )
        except ValueError:
            min_edge = DEFAULT_OPUS_MIN_SONNET_EDGE
        direct_tiny = (
            env_map.get("KALIBRE_OPUS_DIRECT_WHEN_TINY_CANDIDATE_SET", "").strip() == "1"
        )
        try:
            disagree = float(
                env_map.get("KALIBRE_OPUS_DISAGREEMENT_THRESHOLD")
                or DEFAULT_OPUS_DISAGREEMENT_THRESHOLD,
            )
        except ValueError:
            disagree = DEFAULT_OPUS_DISAGREEMENT_THRESHOLD
        return cls(
            enabled=enabled,
            max_calls_per_tick=max(0, max_calls),
            for_exploration=for_exploration,
            min_sonnet_edge=max(0.0, min_edge),
            direct_when_tiny_candidate_set=direct_tiny,
            disagreement_threshold=max(0.0, disagree),
        )


# Edge sources written by the Opus escalation path. Stable strings; new
# values must be added explicitly so analytics splits stay clean.
EDGE_SONNET_OPUS_CONFIRMED = "sonnet_opus_confirmed"
EDGE_OPUS_EXPLORATION = "opus_exploration"
EDGE_OPUS_DIRECT = "opus_direct"
EDGE_MODEL_DISAGREEMENT = "model_disagreement"
# Phase 6B: web-search-enriched Sonnet/Opus edges. Set when the provider
# actually received >= 1 server_tool_use.web_search_request AND emitted
# at least one cited URL. Empty-search forecasts keep the original
# sonnet_forecast / opus_forecast label and gain a ``web_search_empty``
# audit note instead.
EDGE_WEB_SONNET_FORECAST = "web_sonnet_forecast"
EDGE_WEB_OPUS_FORECAST = "web_opus_forecast"
# Phase 6C forecast-repair: when a web-search forecast returns
# parse_error, the runner retries once with web_search_enabled_override=False
# (strict json_object response_format) and surfaces it under these labels
# so analytics can split true non-search forecasts from rescued ones.
EDGE_SONNET_FALLBACK_FORECAST = "sonnet_fallback_forecast"
EDGE_OPUS_FALLBACK_FORECAST = "opus_fallback_forecast"
# Phase 6D budget-aware routing labels. Emitted when the runner stops
# issuing further paid forecast calls because the per-tick LLM spend
# cap was reached, OR when Opus is skipped under the micro budget
# profile. The latter is informational -- it preserves the audit trail
# so reports can count "Opus calls we deliberately did not make".
EDGE_BLOCKED_TICK_BUDGET = "forecast_blocked_tick_budget"
EDGE_OPUS_SKIPPED_BUDGET = "opus_skipped_budget_profile"
# Phase 6E auto-escalation: when the active budget profile says Opus
# should be off (e.g. micro), the runner still escalates to Opus for
# markets where the Sonnet pass produced an actionable edge >=
# ``KALIBRE_AUTO_OPUS_MIN_EDGE_PP`` (default 1.0pp). The candidates
# are restricted to that set rather than the broader exploration /
# disagreement / direct candidate logic. The audit row uses the same
# combine-outcome edge labels (sonnet_opus_confirmed / opus_direct /
# etc.) plus an ``opus_auto_escalated=True`` marker in audit_json.
DEFAULT_AUTO_OPUS_MIN_EDGE_PP = 1.0


def _resolve_sigma(
    *,
    explicit_sigma: float | None,
    model_tier: str | None,
    forecast_config: ForecastConfig,
) -> float:
    """Pick the right sigma for a forecast result and clip into [0.02, 0.25].

    - If the provider returned an explicit ``sigma_p``, honor it (after
      clipping). This is the path used when a model self-reports
      uncertainty.
    - Otherwise, look up the A.3 tier sigma from config:
      ``opus=0.05``, ``sonnet=0.08``, ``haiku=0.12``,
      ``structural_only=0.15``, fallback ``default=0.15``.
    """
    if explicit_sigma is not None:
        return clip_sigma_p(float(explicit_sigma))
    return clip_sigma_p(forecast_config.sigma_for_tier(model_tier))


@dataclass
class ForecastPassResult:
    batch: ForecastBatchResult
    probabilities: dict[str, MarketProbability]
    audit_rows: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    selection: SelectionPassResult | None = None
    # Phase 6B: evidence rows that the loop persists to the new
    # web_search_results / evidence_bundles tables.
    evidence_rows: list[dict[str, Any]] = field(default_factory=list)
    evidence_bundles: list[dict[str, Any]] = field(default_factory=list)
    # Phase 6C repair: one row per evidence-bearing forecast result
    # (Sonnet + Opus), persisted to ``web_search_queries``. OpenRouter
    # doesn't expose individual queries; this is an aggregate per
    # forecast (engine + requests_count).
    web_search_query_rows: list[dict[str, Any]] = field(default_factory=list)
    # Phase 6B: aggregate web-search counters (also surfaced via summary).
    web_search_enabled: bool = False
    web_search_provider: str = "noop"
    search_requests_count: int = 0
    cited_urls_count: int = 0
    evidence_quality_sum: float = 0.0
    evidence_quality_n: int = 0
    web_search_errors: int = 0
    web_search_blocked_spend: bool = False
    web_search_skipped_deadline: bool = False
    # Phase 6D budget-aware routing counters. ``tick_paid_spend_usd`` is
    # the sum of api_cost_usd over fresh (non-cached) provider calls
    # this pass; ``per_tick_paid_forecast_cap_usd`` is the configured
    # ceiling. ``forecast_blocked_tick_budget_count`` counts markets
    # that were skipped because the per-tick cap was already exceeded.
    # ``opus_skipped_budget_count`` counts Opus passes the runner did
    # not make because the budget profile had ``opus_default_on=False``
    # and no ``KALIBRE_FORCE_OPUS=1`` override. ``cache_hit_savings_usd``
    # is a rough estimate of what cache hits would have cost if they
    # had instead gone to the provider.
    budget_profile_name: str = "standard"
    per_tick_paid_forecast_cap_usd: float = 1e9
    tick_paid_spend_usd: float = 0.0
    forecast_blocked_tick_budget_count: int = 0
    opus_skipped_budget_count: int = 0
    cache_hit_savings_usd: float = 0.0

    @property
    def summary(self) -> dict[str, Any]:
        base = self.batch.summary()
        if self.selection is not None:
            sel_summary = self.selection.summary
            base["selected_count"] = sel_summary["selected_count"]
            base["selected_market_ids"] = sel_summary["selected_market_ids"]
            base["selection_skip_reason_counts"] = sel_summary["skip_reason_counts"]
        # Phase 6B summary additions.
        base["web_search_enabled"] = self.web_search_enabled
        base["web_search_provider"] = self.web_search_provider
        base["search_requests_count"] = self.search_requests_count
        base["cited_urls_count"] = self.cited_urls_count
        base["evidence_quality_avg"] = (
            round(self.evidence_quality_sum / self.evidence_quality_n, 4)
            if self.evidence_quality_n else None
        )
        base["web_search_errors"] = self.web_search_errors
        base["web_search_blocked_spend"] = self.web_search_blocked_spend
        base["web_search_skipped_deadline"] = self.web_search_skipped_deadline
        # Phase 6D budget-routing summary fields.
        base["budget_profile"] = self.budget_profile_name
        base["per_tick_paid_forecast_cap_usd"] = (
            round(float(self.per_tick_paid_forecast_cap_usd), 4)
        )
        base["tick_paid_spend_usd"] = round(float(self.tick_paid_spend_usd), 6)
        base["forecast_blocked_tick_budget_count"] = self.forecast_blocked_tick_budget_count
        base["opus_skipped_budget_count"] = self.opus_skipped_budget_count
        base["cache_hit_savings_usd"] = round(float(self.cache_hit_savings_usd), 6)
        return base

    def selection_shadow_rows(self, *, ctx: TickContext) -> list[dict[str, Any]]:
        """Render selection decisions as ``shadow_proposals`` rows."""
        if self.selection is None:
            return []
        return [d.to_shadow_row(ctx=ctx) for d in self.selection.decisions]


def _build_request(market: MarketView) -> ForecastRequest:
    q = market.quote
    return ForecastRequest(
        market_id=market.market_id,
        question=market.question,
        source=market.source,
        topic=market.topic,
        family=market.family,
        resolution_time=market.resolution_time,
        best_bid=q.best_bid or 0.0,
        best_ask=q.best_ask or 0.0,
        spread=q.spread,
        volume_24h=q.volume_24h,
    )


def forecast_audit_row(
    *,
    market_id: str,
    ctx: TickContext,
    edge_source: str,
    decision: str,
    p_raw: float | None = None,
    p_cal: float | None = None,
    sigma_p: float | None = None,
    rationale: str = "",
    api_cost_usd: float = 0.0,
    error: str | None = None,
    model: str = "",
    model_tier: str = "external",
    cache_hit: bool = False,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
    blend: BlendResult | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blend_payload: dict[str, Any] = {}
    if blend is not None:
        blend_payload = dict(blend.to_audit_dict())
    payload: dict[str, Any] = {
        "edge_source": edge_source,
        "tick_ts": ctx.tick_id,
        "version": ctx.version,
        "experiment_id": ctx.experiment_id,
        "participant_idx": ctx.participant_idx,
        "market_id": market_id,
        "decision": decision,
        "model": model,
        "model_tier": model_tier,
        "cache_hit": bool(cache_hit),
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "cached_tokens": int(cached_tokens or 0),
        "p_raw": p_raw,
        "p_cal": p_cal,
        "sigma_p": sigma_p,
        "api_cost_usd": float(api_cost_usd or 0.0),
        "rationale": rationale or "",
        "error": error,
        # Blend fields (p_market, p_blend, market_weight, model_weight,
        # sigma_model, sigma_market, sigma_blend, blend_reason). Always
        # in audit_json so post-hoc analysis can split by regime.
        **blend_payload,
    }
    if extra:
        payload.update(extra)
    return {
        "tick_ts": ctx.tick_id,
        "experiment_id": ctx.experiment_id,
        "participant_idx": ctx.participant_idx,
        "market_id": market_id,
        "model": model,
        "model_tier": model_tier,
        "cache_hit": int(bool(cache_hit)),
        "p_raw": p_raw,
        "p_cal": p_cal,
        "sigma_p": sigma_p,
        "p_market": blend.p_market if blend is not None else None,
        "p_blend": blend.p_blend if blend is not None else None,
        "blend_reason": blend.blend_reason if blend is not None else None,
        "api_cost_usd": float(api_cost_usd or 0.0),
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "cached_tokens": int(cached_tokens or 0),
        "rationale": rationale or "",
        "error": error,
        "edge_source": edge_source,
        "decision": decision,
        "audit_json": json.dumps(payload, default=str),
        "created_at": datetime.now(tz=UTC).isoformat(),
    }


def _blend_for_market(
    *,
    market: MarketView,
    p_model_cal: float,
    sigma_model: float,
    ctx: TickContext,
    optional_extra_shrinkage: float = 0.0,
) -> BlendResult:
    """Compute the blend using the *current* market quote, not any cached snapshot."""
    q = market.quote
    p_market = compute_market_prior(best_bid=q.best_bid, best_ask=q.best_ask)
    return blend_with_market_prior(
        p_model_cal=p_model_cal,
        sigma_model=sigma_model,
        p_market=p_market,
        spread=q.spread,
        volume_24h=q.volume_24h,
        time_to_resolve_hours=market.hours_to_resolution(ctx.now),
        optional_extra_shrinkage=optional_extra_shrinkage,
    )


def _qh_shrinkage(qh_features: dict[str, Any] | None, market_id: str) -> float:
    """Look up the QH deference shrinkage for ``market_id``. Default 0.0."""
    if not qh_features:
        return 0.0
    feat = qh_features.get(market_id)
    if feat is None:
        return 0.0
    value = getattr(feat, "deference_extra_shrinkage", None)
    if value is None:
        return 0.0
    return float(value)


def run_forecast_pass(
    ctx: TickContext,
    *,
    provider: ForecastProvider,
    cache: ForecastCache | None = None,
    calibrator: Calibrator | None = None,
    spend_governor: SpendGovernor | None = None,
    max_markets: int = DEFAULT_MAX_FORECAST_MARKETS,
    deadline_monotonic: float | None = None,
    deadline_skip_buffer_sec: float = DEFAULT_DEADLINE_SKIP_BUFFER_SEC,
    per_call_buffer_sec: float = DEFAULT_PER_CALL_BUFFER_SEC,
    universe_config: UniverseFilterConfig = UniverseFilterConfig(),
    forecast_config: ForecastConfig | None = None,
    cost_estimator: Callable[[str], float] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    cache_ttl_sec: int = DEFAULT_TTL_SEC,
    qh_features: dict[str, Any] | None = None,
    selector: Callable[..., SelectionPassResult] | None = None,
    layers_config: LayersConfig | None = None,
    degraded_mode: str = "full",
    exploration_config: SelectorExplorationConfig | None = None,
    opus_provider: ForecastProvider | None = None,
    opus_config: OpusEscalationConfig | None = None,
    # Phase 6B: OpenRouter native web-search wiring. ``web_search_config``
    # carries the env-driven knobs; ``web_search_spend_today_usd_fn`` is
    # an optional read-only callable returning today's total search
    # spend so the runner can pre-skip when over the daily cap.
    web_search_config: WebSearchConfig | None = None,
    web_search_spend_today_usd_fn: Callable[[], float] | None = None,
    # Phase 6D: BudgetProfile + force-opus flag drive per-tick paid
    # cap, cache TTL, and Opus skip-on-profile. Defaults preserve the
    # pre-6D shape (effectively no cap, Opus runs when its config is on,
    # 30-min web-search cache TTL).
    budget_profile: BudgetProfile | None = None,
    force_opus: bool = False,
) -> ForecastPassResult:
    """Run forecasts for up to ``max_markets`` universe-accepted markets.

    Errors never bubble out: provider failures, parser failures, spend
    blocks, and deadline skips are all converted to audit rows with
    ``edge_source`` reflecting why we held.

    Accounting (``ForecastBatchResult``):

    - ``attempted`` counts every target market processed into a result
      / audit row (cache hits, spend blocks, deadline skips, provider
      errors, and successes). It always equals ``len(batch.results)``
      after the pass.
    - ``succeeded`` counts cache hits plus fresh provider successes.
    - ``cached`` counts cache hits only.
    - ``failed`` counts spend/deadline/provider errors.
    """
    cache = cache if cache is not None else MemoryForecastCache()
    calibrator = calibrator if calibrator is not None else Calibrator(method="identity")
    forecast_cfg = forecast_config or get_default_config()
    profile = budget_profile or BudgetProfile.standard()
    per_tick_paid_cap = float(profile.per_tick_paid_forecast_cap_usd)
    tick_paid_spend = 0.0
    blocked_tick_budget_count = 0
    cache_hit_savings_usd = 0.0
    model = getattr(provider, "model", "unknown")
    provider_tier = _tier_for_governor(provider, None)
    accepted, _decisions = filter_universe(ctx.markets, now=ctx.now, config=universe_config)
    # Phase 4B: deterministic target selector. ``passthrough_selector`` is
    # used as the safe fallback when callers explicitly opt out (legacy
    # tests). Production callers pass ``selector=select_forecast_targets``
    # so the Sonnet budget lands on tradable edges, not the first N.
    selector_fn = selector or passthrough_selector
    selector_kwargs: dict[str, Any] = dict(
        ctx=ctx,
        markets=accepted,
        max_markets=int(max_markets),
        portfolio=ctx.portfolio,
        qh_features=qh_features,
        layer_config=layers_config,
        degraded_mode=degraded_mode,
    )
    # Phase 6: pass the exploration config through if the production
    # selector accepts it. ``passthrough_selector`` swallows kwargs.
    if exploration_config is not None:
        selector_kwargs["exploration_config"] = exploration_config
    try:
        selection = selector_fn(**selector_kwargs)
    except TypeError:
        # Legacy selector that pre-dates ``exploration_config``.
        selector_kwargs.pop("exploration_config", None)
        selection = selector_fn(**selector_kwargs)
    targets = list(selection.selected)
    # Phase 6: TODO(phase6-followup): family-pack JSON-map mode would
    # batch multiple markets in the same family into one prompt; see
    # Guide.md §18 (Known limitations). Per-market forecasting below is
    # unchanged.

    batch = ForecastBatchResult(model=model)
    audit_rows: list[dict[str, Any]] = []
    probabilities: dict[str, MarketProbability] = {}
    notes: list[str] = []
    # Phase 6B aggregate counters (rolled into ForecastPassResult below).
    web_search_enabled = bool(web_search_config and web_search_config.enabled)
    web_search_provider_label = (
        f"openrouter:web_search:{web_search_config.engine}"
        if web_search_enabled else "noop"
    )
    web_search_blocked_spend = False
    web_search_skipped_deadline = False
    web_search_blocked_thin_universe = False
    # Phase 6C repair: when the universe is too thin to justify a search
    # round-trip, force-disable web search for the per-market provider
    # calls. The provider's ProviderConfig is unchanged; we pass
    # ``web_search_enabled_override=False`` at call time so no ``tools``
    # array ships and no search cost is incurred.
    if (
        web_search_enabled
        and web_search_config is not None
        and web_search_config.min_universe_for_search > 0
        and len(accepted) <= web_search_config.min_universe_for_search
    ):
        web_search_blocked_thin_universe = True
        web_search_enabled = False
        notes.append(
            f"web_search_thin_universe:{len(accepted)}<={web_search_config.min_universe_for_search}"
        )
        web_search_provider_label = "noop:thin_universe"
    # Pre-call: when search is active, check today's search spend cap.
    # Phase 6C repair: on block we ACTUALLY disable tools for this pass
    # (per-call override), not merely change labels.
    if web_search_enabled and web_search_spend_today_usd_fn is not None:
        try:
            spent_today = float(web_search_spend_today_usd_fn() or 0.0)
        except Exception:
            spent_today = 0.0
        if spent_today >= float(web_search_config.daily_hard_cap_usd):
            web_search_blocked_spend = True
            web_search_enabled = False  # demote for this pass
            notes.append("web_search_blocked_spend")
            web_search_provider_label = "noop:spend_capped"
    # Phase 6B: stamp the detected category onto every selected request
    # so the prompt's category playbook block fires.
    if targets:
        try:
            from kalibre.categories import detect_category
        except Exception:
            detect_category = None  # type: ignore[assignment]
    else:
        detect_category = None  # type: ignore[assignment]
    request_overrides: dict[str, ForecastRequest] = {}
    if detect_category is not None:
        from dataclasses import replace as _replace
        for market in targets:
            category_profile = detect_category(
                market.question, market.topic, market.family, market.source,
            )
            req = _build_request(market)
            request_overrides[market.market_id] = _replace(
                req, category=category_profile.category,
            )

    if not targets:
        notes.append("no_universe_candidates_for_forecast")
        return ForecastPassResult(
            batch=batch, probabilities=probabilities, audit_rows=audit_rows, notes=notes,
            selection=selection,
            web_search_enabled=web_search_enabled,
            web_search_provider=web_search_provider_label,
            web_search_blocked_spend=web_search_blocked_spend,
            web_search_skipped_deadline=web_search_skipped_deadline,
            budget_profile_name=profile.name,
            per_tick_paid_forecast_cap_usd=per_tick_paid_cap,
        )

    # Phase 6B: when web search is active, a single forecast call needs
    # noticeably more headroom (the model issues a server-side search).
    # Compute an effective skip buffer that takes the larger of the
    # legacy buffer and the web-search buffer.
    effective_skip_buffer = deadline_skip_buffer_sec
    if web_search_enabled and web_search_config is not None:
        effective_skip_buffer = max(
            deadline_skip_buffer_sec, float(web_search_config.deadline_buffer_sec),
        )
    # Deadline pre-check: skip every call if we don't have headroom.
    if (
        deadline_monotonic is not None
        and (deadline_monotonic - monotonic()) < effective_skip_buffer
    ):
        if web_search_enabled:
            notes.append("web_search_skipped_deadline")
            web_search_skipped_deadline = True
        notes.append("deadline_skip_all_forecasts")
        for market in targets:
            batch.attempted += 1
            batch.failed += 1
            batch.results.append(ForecastResult(
                market_id=market.market_id, error="deadline_near",
                edge_source="forecast_blocked_deadline", model=model,
                model_tier=provider_tier,
            ))
            audit_rows.append(forecast_audit_row(
                market_id=market.market_id,
                ctx=ctx,
                edge_source="forecast_blocked_deadline",
                decision="skip",
                error="deadline_near",
                model=model,
                model_tier=provider_tier,
            ))
        return ForecastPassResult(
            batch=batch, probabilities=probabilities, audit_rows=audit_rows, notes=notes,
            selection=selection,
            web_search_enabled=web_search_enabled,
            web_search_provider=web_search_provider_label,
            web_search_blocked_spend=web_search_blocked_spend,
            web_search_skipped_deadline=web_search_skipped_deadline,
            budget_profile_name=profile.name,
            per_tick_paid_forecast_cap_usd=per_tick_paid_cap,
        )

    if cost_estimator is not None:
        estimated_cost_each = cost_estimator(model)
    else:
        estimated_cost_each = estimate_cost_usd(model, config=forecast_cfg)
    if isinstance(provider, NoOpForecastProvider):
        # The runner still goes through the audit path so the operator sees why.
        pass

    for market in targets:
        batch.attempted += 1
        request = request_overrides.get(market.market_id) or _build_request(market)
        # Phase 6C: cache web-search forecasts across ticks within a
        # bounded TTL. The cache key includes ``web_search_enabled`` +
        # engine so pre-6B rows never satisfy a 6B query, and we
        # deliberately omit ``evidence_hash`` from the lookup key (which
        # we couldn't know pre-call anyway) so a fresh call simply
        # populates the entry and subsequent ticks reuse it. Lower TTL
        # for search-bearing rows because evidence freshness matters.
        if web_search_enabled:
            key = cache_key(
                request, model,
                web_search_enabled=True,
                web_search_engine=(web_search_config.engine if web_search_config else None),
            )
            # Phase 6D: under the micro profile the web-search cache
            # TTL goes from the legacy 30-min cap to 2h, with a fallback
            # to ``cache_ttl_short_resolution_sec`` for markets that
            # resolve within ``short_resolution_threshold_hours``. The
            # short-resolution path keeps evidence fresh near kickoff
            # where injury / lineup news still matters.
            short_path = False
            try:
                hours_to_resolve = market.hours_to_resolution(ctx.now)
            except Exception:
                hours_to_resolve = None
            if (
                hours_to_resolve is not None
                and hours_to_resolve <= profile.short_resolution_threshold_hours
            ):
                short_path = True
            web_ttl = (
                int(profile.cache_ttl_short_resolution_sec)
                if short_path else int(profile.cache_ttl_sec)
            )
            effective_ttl = min(int(cache_ttl_sec), web_ttl)
            cached = cache.get(key, ttl_sec=effective_ttl)
        else:
            key = cache_key(request, model)
            cached = cache.get(key, ttl_sec=cache_ttl_sec)
        if cached is not None:
            batch.cached += 1
            batch.succeeded += 1
            # Phase 6D: cache hits avoid a fresh provider call. Estimate
            # the savings using the per-call cost estimate so the report
            # can surface "we saved $X this tick by caching".
            cache_hit_savings_usd += float(estimated_cost_each)
            p_cal = calibrate_probability(cached.p_raw, calibrator)
            cached.p_mean = p_cal
            cached_tier = cached.model_tier or provider_tier
            cached_sigma = _resolve_sigma(
                explicit_sigma=cached.sigma_p,
                model_tier=cached_tier,
                forecast_config=forecast_cfg,
            )
            cached.sigma_p = cached_sigma
            cached.api_cost_usd = 0.0  # never re-charge on hit
            # Blend recomputed from the *current* market quote, not the
            # cached request's quote. Cache replays the model output only.
            blend = _blend_for_market(
                market=market,
                p_model_cal=p_cal,
                sigma_model=cached_sigma,
                ctx=ctx,
                optional_extra_shrinkage=_qh_shrinkage(qh_features, market.market_id),
            )
            batch.results.append(cached)
            probabilities[market.market_id] = MarketProbability(
                p_mean=blend.p_blend,
                sigma_p=blend.sigma_blend,
                edge_source=cached.edge_source or "sonnet_forecast",
                model_tier=cached_tier,
                # Phase 6R: preserve the un-blended model probability so
                # a later Opus combine can avoid double-counting the
                # market prior.
                p_model_only=float(p_cal),
                sigma_model_only=float(cached_sigma),
            )
            audit_rows.append(forecast_audit_row(
                market_id=market.market_id,
                ctx=ctx,
                edge_source=cached.edge_source or "sonnet_forecast",
                decision="cached",
                p_raw=cached.p_raw,
                p_cal=p_cal,
                sigma_p=cached_sigma,
                rationale=cached.rationale,
                api_cost_usd=0.0,
                model=cached.model or model,
                model_tier=cached_tier,
                cache_hit=True,
                input_tokens=cached.input_tokens,
                output_tokens=cached.output_tokens,
                cached_tokens=cached.cached_tokens,
                blend=blend,
                # Phase 6C polish: replay the Phase 6B evidence fields on
                # cache hits so audit_json downstream readers (the
                # evidence_report counter, forecasts table queries) see
                # cited_urls + evidence_quality + evidence_hash etc. just
                # like on the fresh-call path. ``cache_hit=true`` is
                # already on the row so reports can split fresh vs cached.
                extra={
                    "category": cached.category,
                    "evidence_quality": cached.evidence_quality,
                    "evidence_hash": cached.evidence_hash,
                    "cited_urls": list(cached.cited_urls or ()),
                    "key_drivers_json": cached.key_drivers_json,
                    "stale_evidence": cached.stale_evidence,
                    "web_search_requests": int(cached.web_search_requests or 0),
                    "web_search_enabled": web_search_enabled,
                },
            ))
            continue

        # Spend gate.
        if spend_governor is not None:
            decision = spend_governor.can_call(provider_tier, estimated_cost_each)
            if not decision.allowed:
                batch.failed += 1
                batch.results.append(ForecastResult(
                    market_id=market.market_id,
                    error=decision.reason,
                    edge_source="forecast_blocked_spend",
                    model=model,
                    model_tier=provider_tier,
                ))
                audit_rows.append(forecast_audit_row(
                    market_id=market.market_id,
                    ctx=ctx,
                    edge_source="forecast_blocked_spend",
                    decision="skip",
                    error=decision.reason,
                    model=model,
                    model_tier=provider_tier,
                ))
                continue

        # Phase 6D budget-aware routing: per-tick paid forecast cap.
        # Skip when we would breach the cap with this call. ``tick_paid_spend``
        # tracks only fresh (non-cache-hit) provider spend within this
        # pass. Cache hits are free and pass through without checking.
        if (
            per_tick_paid_cap > 0
            and tick_paid_spend + estimated_cost_each > per_tick_paid_cap
        ):
            batch.failed += 1
            blocked_tick_budget_count += 1
            batch.results.append(ForecastResult(
                market_id=market.market_id,
                error="tick_paid_forecast_cap_reached",
                edge_source=EDGE_BLOCKED_TICK_BUDGET,
                model=model,
                model_tier=provider_tier,
            ))
            audit_rows.append(forecast_audit_row(
                market_id=market.market_id,
                ctx=ctx,
                edge_source=EDGE_BLOCKED_TICK_BUDGET,
                decision="skip",
                error="tick_paid_forecast_cap_reached",
                model=model,
                model_tier=provider_tier,
                extra={
                    "budget_profile": profile.name,
                    "per_tick_paid_forecast_cap_usd": per_tick_paid_cap,
                    "tick_paid_spend_usd_so_far": round(tick_paid_spend, 6),
                    "estimated_cost_each": round(estimated_cost_each, 6),
                },
            ))
            continue

        # Per-call deadline gate.
        if (
            deadline_monotonic is not None
            and (deadline_monotonic - monotonic()) < per_call_buffer_sec
        ):
            batch.failed += 1
            batch.results.append(ForecastResult(
                market_id=market.market_id,
                error="deadline_near",
                edge_source="forecast_blocked_deadline",
                model=model,
                model_tier=provider_tier,
            ))
            audit_rows.append(forecast_audit_row(
                market_id=market.market_id,
                ctx=ctx,
                edge_source="forecast_blocked_deadline",
                decision="skip",
                error="deadline_near",
                model=model,
                model_tier=provider_tier,
            ))
            continue

        # Issue the call. Phase 6C repair: pass an explicit per-call
        # web_search override so the runner-side "blocked" state
        # (thin universe, spend cap, deadline) ACTUALLY suppresses the
        # ``tools`` array in the request body rather than only re-labelling.
        # Some legacy fakes don't accept the new kwarg; fall back to the
        # positional-only call shape if so.
        try:
            result = provider.forecast_one(
                request, web_search_enabled_override=web_search_enabled,
            )
        except TypeError:
            result = provider.forecast_one(request)
        if result.error is not None or result.p_raw is None:
            # Phase 6C forecast-repair: a parse_error on a web-search
            # forecast keeps full spend + cited URLs (the API call ran)
            # but no probability. Try one strict-JSON fallback with web
            # search OFF if enough headroom remains. The original error
            # row is preserved for audit + spend reconciliation; the
            # fallback row, if it succeeds, becomes the usable
            # probability. If the fallback also errors, both rows persist
            # and the market stays missing_probability downstream.
            error_str = result.error or "unknown_error"
            is_parse_error = error_str.startswith("parse_error")
            fallback_eligible = (
                web_search_enabled
                and is_parse_error
                and (
                    deadline_monotonic is None
                    or (deadline_monotonic - monotonic()) >= per_call_buffer_sec
                )
            )
            batch.failed += 1
            batch.results.append(result)
            # Phase 6D: a parse_error still incurred a real API call --
            # count its api_cost_usd against the per-tick paid cap so
            # downstream reports/governance reflect actual spend.
            tick_paid_spend += float(result.api_cost_usd or 0.0)
            audit_rows.append(forecast_audit_row(
                market_id=market.market_id,
                ctx=ctx,
                edge_source=result.edge_source or "forecast_error",
                decision="error",
                error=error_str,
                api_cost_usd=result.api_cost_usd,
                model=result.model or model,
                model_tier=result.model_tier or provider_tier,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cached_tokens=result.cached_tokens,
                extra={
                    # Preserve the evidence the failing call actually
                    # retrieved so spend / cited-URL reports are honest.
                    "cited_urls": list(result.cited_urls or ()),
                    "web_search_requests": int(result.web_search_requests or 0),
                    "web_search_enabled": web_search_enabled,
                    "fallback_attempted": bool(fallback_eligible),
                },
            ))
            if not fallback_eligible:
                continue
            # Strict-JSON fallback call. Spend governor + cache key both
            # use the non-search shape, so a successful fallback caches
            # under the legacy key and reuses across ticks naturally.
            notes.append(f"web_search_parse_error_fallback:{market.market_id}")
            if spend_governor is not None:
                fb_decision = spend_governor.can_call(provider_tier, estimated_cost_each)
                if not fb_decision.allowed:
                    audit_rows.append(forecast_audit_row(
                        market_id=market.market_id,
                        ctx=ctx,
                        edge_source="forecast_blocked_spend",
                        decision="skip",
                        error=fb_decision.reason,
                        model=model,
                        model_tier=provider_tier,
                        extra={"fallback_from_web_parse_error": True},
                    ))
                    continue
            try:
                fb_result = provider.forecast_one(
                    request, web_search_enabled_override=False,
                )
            except TypeError:
                fb_result = provider.forecast_one(request)
            if fb_result.error is not None or fb_result.p_raw is None:
                batch.attempted += 1
                batch.failed += 1
                batch.results.append(fb_result)
                audit_rows.append(forecast_audit_row(
                    market_id=market.market_id,
                    ctx=ctx,
                    edge_source=fb_result.edge_source or "forecast_error",
                    decision="error",
                    error=fb_result.error or "unknown_error",
                    api_cost_usd=fb_result.api_cost_usd,
                    model=fb_result.model or model,
                    model_tier=fb_result.model_tier or provider_tier,
                    input_tokens=fb_result.input_tokens,
                    output_tokens=fb_result.output_tokens,
                    cached_tokens=fb_result.cached_tokens,
                    extra={"fallback_from_web_parse_error": True},
                ))
                continue
            fb_tier = fb_result.model_tier or provider_tier
            fb_result.model_tier = fb_tier
            fb_result.sigma_p = _resolve_sigma(
                explicit_sigma=fb_result.sigma_p,
                model_tier=fb_tier,
                forecast_config=forecast_cfg,
            )
            fb_p_cal = calibrate_probability(fb_result.p_raw, calibrator)
            fb_result.p_mean = fb_p_cal
            if request.category and not fb_result.category:
                fb_result.category = request.category
            if fb_tier == "opus":
                fb_result.edge_source = EDGE_OPUS_FALLBACK_FORECAST
            else:
                fb_result.edge_source = EDGE_SONNET_FALLBACK_FORECAST
            # Strict-json fallback cache key uses the pre-6B (search-off)
            # shape so reuses across ticks naturally.
            fb_key = cache_key(request, model)
            cache.put(fb_key, fb_result)
            if spend_governor is not None and fb_result.api_cost_usd > 0:
                try:
                    spend_governor.record(fb_result.api_cost_usd)
                except ValueError:
                    pass
            # Phase 6D: fallback's fresh call counts toward the per-tick
            # paid cap; the failed-original row's cost was already
            # tracked when its provider call returned (above).
            tick_paid_spend += float(fb_result.api_cost_usd or 0.0)
            fb_blend = _blend_for_market(
                market=market,
                p_model_cal=fb_p_cal,
                sigma_model=fb_result.sigma_p,
                ctx=ctx,
                optional_extra_shrinkage=_qh_shrinkage(qh_features, market.market_id),
            )
            batch.attempted += 1
            batch.succeeded += 1
            batch.spend_estimate_usd += fb_result.api_cost_usd
            batch.results.append(fb_result)
            probabilities[market.market_id] = MarketProbability(
                p_mean=fb_blend.p_blend,
                sigma_p=fb_blend.sigma_blend,
                edge_source=fb_result.edge_source,
                model_tier=fb_tier,
                p_model_only=float(fb_p_cal),
                sigma_model_only=float(fb_result.sigma_p),
            )
            audit_rows.append(forecast_audit_row(
                market_id=market.market_id,
                ctx=ctx,
                edge_source=fb_result.edge_source,
                decision="ok",
                p_raw=fb_result.p_raw,
                p_cal=fb_p_cal,
                sigma_p=fb_result.sigma_p,
                rationale=fb_result.rationale,
                api_cost_usd=fb_result.api_cost_usd,
                model=fb_result.model or model,
                model_tier=fb_tier,
                input_tokens=fb_result.input_tokens,
                output_tokens=fb_result.output_tokens,
                cached_tokens=fb_result.cached_tokens,
                blend=fb_blend,
                extra={
                    "category": fb_result.category,
                    "fallback_from_web_parse_error": True,
                    "web_search_enabled": False,
                    "web_search_requests": 0,
                },
            ))
            continue

        resolved_tier = result.model_tier or provider_tier
        result.model_tier = resolved_tier
        result.sigma_p = _resolve_sigma(
            explicit_sigma=result.sigma_p,
            model_tier=resolved_tier,
            forecast_config=forecast_cfg,
        )
        p_cal = calibrate_probability(result.p_raw, calibrator)
        result.p_mean = p_cal
        # Phase 6B: stamp the runner-side category onto the result so
        # downstream audit + persistence stays uniform with the request.
        if request.category and not result.category:
            result.category = request.category
        # Phase 6B: derive evidence_hash for the cache key / audit, then
        # rewrite the edge_source label when the provider actually used
        # web search.
        if web_search_enabled:
            ev_hash = compute_evidence_hash(
                cited_urls=result.cited_urls,
                web_search_requests=result.web_search_requests,
            )
            result.evidence_hash = ev_hash
            # Phase 6B: presence of cited URLs is the canonical signal
            # that web evidence reached the prompt. Some engines don't
            # populate ``usage.server_tool_use.web_search_requests`` even
            # when they cite, so treat citations alone as enough to
            # promote the edge_source label.
            if result.cited_urls:
                if resolved_tier == "opus":
                    result.edge_source = EDGE_WEB_OPUS_FORECAST
                else:
                    result.edge_source = EDGE_WEB_SONNET_FORECAST
            else:
                notes.append(f"web_search_empty:{market.market_id}")
        # Cache stores the raw post-provider result (without cache_hit flip).
        # The blend is recomputed every tick from the current quote, so we
        # deliberately do NOT cache the blended probability.
        cache.put(key, result)

        if spend_governor is not None and result.api_cost_usd > 0:
            try:
                spend_governor.record(result.api_cost_usd)
            except ValueError:
                pass
        # Phase 6D: this fresh provider call adds to the per-tick paid
        # spend. Cache hits skip this branch entirely.
        tick_paid_spend += float(result.api_cost_usd or 0.0)
        blend = _blend_for_market(
            market=market,
            p_model_cal=p_cal,
            sigma_model=result.sigma_p,
            ctx=ctx,
            optional_extra_shrinkage=_qh_shrinkage(qh_features, market.market_id),
        )
        batch.succeeded += 1
        batch.spend_estimate_usd += result.api_cost_usd
        batch.results.append(result)
        probabilities[market.market_id] = MarketProbability(
            p_mean=blend.p_blend,
            sigma_p=blend.sigma_blend,
            edge_source=result.edge_source or "sonnet_forecast",
            model_tier=resolved_tier,
            # Phase 6R: preserve the un-blended model probability so a
            # later Opus combine can avoid double-counting the market
            # prior.
            p_model_only=float(p_cal),
            sigma_model_only=float(result.sigma_p),
        )
        audit_rows.append(forecast_audit_row(
            market_id=market.market_id,
            ctx=ctx,
            edge_source=result.edge_source or "sonnet_forecast",
            decision="ok",
            p_raw=result.p_raw,
            p_cal=p_cal,
            sigma_p=result.sigma_p,
            rationale=result.rationale,
            api_cost_usd=result.api_cost_usd,
            model=result.model or model,
            model_tier=result.model_tier or "sonnet",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cached_tokens=result.cached_tokens,
            blend=blend,
            extra={
                # Phase 6B audit extras (always present; defaults are
                # None/0/False for non-search forecasts).
                "category": result.category,
                "evidence_quality": result.evidence_quality,
                "evidence_hash": result.evidence_hash,
                "cited_urls": list(result.cited_urls or ()),
                "key_drivers_json": result.key_drivers_json,
                "stale_evidence": result.stale_evidence,
                "web_search_requests": int(result.web_search_requests or 0),
                "web_search_enabled": web_search_enabled,
                # Phase 6D budget-routing extras so the funnel/evidence
                # reports can pull the active profile + per-tick cap
                # from the most-recent audit_json.
                "budget_profile": profile.name,
                "per_tick_paid_forecast_cap_usd": per_tick_paid_cap,
            },
        ))

    if isinstance(provider, NoOpForecastProvider):
        notes.append("provider:noop")

    # --- Phase 6: Opus escalation pass ------------------------------------
    # Phase 6C repair: Opus runs BEFORE evidence build so any Opus
    # citations are persisted to web_search_results / evidence_bundles
    # alongside Sonnet's.
    opus_skipped_budget_count = 0
    # Phase 6D / 6E budget-aware Opus gate.
    #
    # Three regimes:
    #   1. ``opus_allowed_by_profile`` -> standard ``_run_opus_escalation``
    #      (full candidate set: exploration / disagreement / direct).
    #   2. NOT allowed AND Sonnet produced an actionable edge on at
    #      least one market -> Phase 6E auto-escalation: run
    #      ``_run_opus_escalation`` restricted to those markets so we
    #      confirm trade signals before they reach sizing.
    #   3. NOT allowed AND no actionable edge -> skip Opus, emit one
    #      ``opus_skipped_budget_profile`` audit row per market the
    #      regular candidate logic would have picked. This preserves
    #      the audit trail without spending.
    opus_allowed_by_profile = profile.opus_default_on or force_opus
    try:
        auto_opus_min_edge_pp = float(
            os.environ.get("KALIBRE_AUTO_OPUS_MIN_EDGE_PP")
            or DEFAULT_AUTO_OPUS_MIN_EDGE_PP,
        )
    except ValueError:
        auto_opus_min_edge_pp = DEFAULT_AUTO_OPUS_MIN_EDGE_PP
    auto_opus_market_ids: list[str] = []
    if (
        opus_config is not None
        and opus_config.enabled
        and opus_provider is not None
        and opus_config.max_calls_per_tick > 0
        and not opus_allowed_by_profile
    ):
        auto_opus_market_ids = _compute_auto_opus_market_ids(
            targets=targets,
            probabilities=probabilities,
            threshold_pp=auto_opus_min_edge_pp,
        )
    if (
        opus_config is not None
        and opus_config.enabled
        and opus_provider is not None
        and opus_config.max_calls_per_tick > 0
        and not opus_allowed_by_profile
        and not auto_opus_market_ids
    ):
        # Regime 3: profile says no, Sonnet didn't disagree enough.
        candidate_ids = _opus_candidate_market_ids(
            targets=targets,
            selection=selection,
            probabilities=probabilities,
            opus_config=opus_config,
        )
        for mid in candidate_ids[: opus_config.max_calls_per_tick]:
            opus_skipped_budget_count += 1
            notes.append(f"opus_skipped_budget_profile:{mid}")
            audit_rows.append(forecast_audit_row(
                market_id=mid,
                ctx=ctx,
                edge_source=EDGE_OPUS_SKIPPED_BUDGET,
                decision="skip",
                error=f"opus_skipped_budget_profile:{profile.name}",
                model=getattr(opus_provider, "model", "opus"),
                model_tier="opus",
                extra={
                    "budget_profile": profile.name,
                    "force_opus": bool(force_opus),
                    "reason": "opus_default_off_for_profile",
                    "auto_opus_min_edge_pp": auto_opus_min_edge_pp,
                },
            ))
    elif (
        opus_config is not None
        and opus_config.enabled
        and opus_provider is not None
        and opus_config.max_calls_per_tick > 0
        and not opus_allowed_by_profile
        and auto_opus_market_ids
    ):
        # Regime 2: auto-escalation. Restrict the Opus candidates to
        # markets where the Sonnet blend produced an actionable edge.
        notes.append(
            f"opus_auto_escalated:{','.join(auto_opus_market_ids)}"
        )
        _run_opus_escalation(
            ctx=ctx,
            targets=targets,
            selection=selection,
            probabilities=probabilities,
            batch=batch,
            audit_rows=audit_rows,
            notes=notes,
            opus_provider=opus_provider,
            opus_config=opus_config,
            spend_governor=spend_governor,
            forecast_cfg=forecast_cfg,
            calibrator=calibrator,
            cost_estimator=cost_estimator,
            cache_ttl_sec=cache_ttl_sec,
            cache=cache,
            monotonic=monotonic,
            deadline_monotonic=deadline_monotonic,
            per_call_buffer_sec=per_call_buffer_sec,
            qh_features=qh_features,
            web_search_enabled=web_search_enabled,
            request_overrides=request_overrides,
            restrict_to_market_ids=auto_opus_market_ids,
            auto_escalated=True,
        )
    elif (
        opus_config is not None
        and opus_config.enabled
        and opus_provider is not None
        and opus_config.max_calls_per_tick > 0
    ):
        _run_opus_escalation(
            ctx=ctx,
            targets=targets,
            selection=selection,
            probabilities=probabilities,
            batch=batch,
            audit_rows=audit_rows,
            notes=notes,
            opus_provider=opus_provider,
            opus_config=opus_config,
            spend_governor=spend_governor,
            forecast_cfg=forecast_cfg,
            calibrator=calibrator,
            cost_estimator=cost_estimator,
            cache_ttl_sec=cache_ttl_sec,
            cache=cache,
            monotonic=monotonic,
            deadline_monotonic=deadline_monotonic,
            per_call_buffer_sec=per_call_buffer_sec,
            qh_features=qh_features,
            web_search_enabled=web_search_enabled,
            request_overrides=request_overrides,
        )

    # Phase 6C repair: build evidence + bundle rows from ALL successful
    # results (Sonnet + Opus). Also emit one ``web_search_queries`` row
    # per forecast that has evidence (citations OR web_search_requests
    # > 0). Counter logic uses citations-or-requests rather than the
    # legacy ``web_search_requests > 0`` heuristic so engines that don't
    # populate ``server_tool_use`` (the inferred-citations path in
    # provider.py) still get counted as evidence-bearing.
    evidence_rows: list[dict[str, Any]] = []
    evidence_bundles: list[dict[str, Any]] = []
    web_search_query_rows: list[dict[str, Any]] = []
    search_requests_count = 0
    cited_urls_count = 0
    evidence_quality_sum = 0.0
    evidence_quality_n = 0
    web_search_errors = 0
    if web_search_enabled:
        try:
            from kalibre.categories import classify_source_tier
        except Exception:
            classify_source_tier = lambda d: "D"  # type: ignore[assignment]
        from urllib.parse import urlparse as _urlparse
        from datetime import UTC as _UTC, datetime as _datetime
        created = _datetime.now(tz=_UTC).isoformat()
        engine_label = (
            web_search_config.engine
            if web_search_config is not None else None
        )
        # Phase 6C repair: dedup by market_id so a Sonnet + Opus pair on
        # the same market produces one bundle row (urls merged).
        bundles_by_mid: dict[str, dict[str, Any]] = {}
        for fr in batch.results:
            requests_n = int(fr.web_search_requests or 0)
            urls = list(fr.cited_urls or ())
            has_evidence = (requests_n > 0) or bool(urls)
            if fr.error is not None:
                if requests_n:
                    search_requests_count += requests_n
                continue
            if not has_evidence:
                # Forecast succeeded but the model didn't use the tool;
                # no evidence rows for this one.
                continue
            # Phase 6C polish: a cache hit replays evidence the prior
            # tick already paid for and persisted. Do NOT emit fresh
            # evidence_rows / evidence_bundles / web_search_queries on
            # cache hits and do NOT roll cached searches into the
            # fresh-search counter -- otherwise reports would
            # double-count search activity and inflate the daily spend
            # picture. The cache-hit forecast row itself still carries
            # cited_urls via audit_json (above), so evidence_report's
            # per-forecast accounting picks it up correctly.
            if fr.cache_hit:
                continue
            search_requests_count += requests_n
            cited_urls_count += len(urls)
            if fr.evidence_quality is not None:
                evidence_quality_sum += float(fr.evidence_quality)
                evidence_quality_n += 1
            if requests_n > 0 and not urls:
                web_search_errors += 1
            for url in urls:
                domain = ""
                try:
                    domain = _urlparse(url).netloc.lower()
                except Exception:
                    domain = ""
                evidence_rows.append({
                    "tick_ts": ctx.tick_id,
                    "experiment_id": ctx.experiment_id,
                    "market_id": fr.market_id,
                    "url": url,
                    "domain": domain,
                    "source_tier": classify_source_tier(domain),
                    "title": "",
                    "snippet": "",
                    "created_at": created,
                })
            # Merge bundle rows per (tick, market) so dual Sonnet+Opus
            # forecasts produce one bundle with combined URL list.
            existing = bundles_by_mid.get(fr.market_id)
            if existing is None:
                bundles_by_mid[fr.market_id] = {
                    "tick_ts": ctx.tick_id,
                    "experiment_id": ctx.experiment_id,
                    "market_id": fr.market_id,
                    "evidence_hash": fr.evidence_hash or "",
                    "urls_json": json.dumps(urls, default=str),
                    "evidence_quality": (
                        float(fr.evidence_quality)
                        if fr.evidence_quality is not None else None
                    ),
                    "stale_evidence": (
                        int(bool(fr.stale_evidence))
                        if fr.stale_evidence is not None else None
                    ),
                    "key_drivers_json": fr.key_drivers_json,
                    "web_search_requests": requests_n,
                    "created_at": created,
                }
            else:
                # Merge URL lists (dedup preserving order) and take the
                # max evidence_quality across the two forecasts.
                try:
                    prior = json.loads(existing["urls_json"] or "[]")
                except (TypeError, ValueError):
                    prior = []
                merged_seen = list(dict.fromkeys(list(prior) + urls))
                existing["urls_json"] = json.dumps(merged_seen, default=str)
                existing["web_search_requests"] = (
                    int(existing.get("web_search_requests") or 0) + requests_n
                )
                if fr.evidence_quality is not None:
                    prev_q = existing.get("evidence_quality")
                    existing["evidence_quality"] = (
                        float(fr.evidence_quality)
                        if prev_q is None
                        else max(float(prev_q), float(fr.evidence_quality))
                    )
            # One ``web_search_queries`` row per evidence-bearing
            # forecast result. OpenRouter doesn't expose the actual
            # queries, so this is an aggregate-per-forecast record.
            web_search_query_rows.append({
                "tick_ts": ctx.tick_id,
                "experiment_id": ctx.experiment_id,
                "market_id": fr.market_id,
                "engine": engine_label,
                "requests_count": requests_n if requests_n > 0 else len(urls),
                "created_at": created,
            })
        evidence_bundles = list(bundles_by_mid.values())

    return ForecastPassResult(
        batch=batch, probabilities=probabilities, audit_rows=audit_rows, notes=notes,
        selection=selection,
        evidence_rows=evidence_rows,
        evidence_bundles=evidence_bundles,
        web_search_query_rows=web_search_query_rows,
        web_search_enabled=web_search_enabled,
        web_search_provider=web_search_provider_label,
        search_requests_count=search_requests_count,
        cited_urls_count=cited_urls_count,
        evidence_quality_sum=evidence_quality_sum,
        evidence_quality_n=evidence_quality_n,
        web_search_errors=web_search_errors,
        web_search_blocked_spend=web_search_blocked_spend,
        web_search_skipped_deadline=web_search_skipped_deadline,
        # Phase 6D budget-routing fields.
        budget_profile_name=profile.name,
        per_tick_paid_forecast_cap_usd=per_tick_paid_cap,
        tick_paid_spend_usd=tick_paid_spend,
        forecast_blocked_tick_budget_count=blocked_tick_budget_count,
        opus_skipped_budget_count=opus_skipped_budget_count,
        cache_hit_savings_usd=cache_hit_savings_usd,
    )


# --- Phase 6: Opus escalation helpers -------------------------------------


def _compute_auto_opus_market_ids(
    *,
    targets: list[MarketView],
    probabilities: dict[str, MarketProbability],
    threshold_pp: float = DEFAULT_AUTO_OPUS_MIN_EDGE_PP,
) -> list[str]:
    """Phase 6E auto-escalation gate.

    Return the market_ids whose Sonnet/blended probability disagrees
    with the market mid by at least ``threshold_pp`` percentage points.
    Used when the budget profile defaults Opus off (e.g. micro) -- in
    that mode the runner still wants to confirm a Sonnet trade signal
    with Opus before letting it through to sizing.

    The gate uses the raw model edge ``|p_mean - market_mid| * 100``
    rather than the side-aware ``(p_eff - side_price)`` rescue edge
    because the strategy hasn't run yet at this point so ``p_eff``
    isn't available. Empirically the two differ by at most the
    half-spread (~0.5-1.0pp); a tighter threshold here means we
    catch *most* would-trade candidates without paying Opus for
    on-mid forecasts.

    Markets without a valid (bid, ask) quote are skipped silently.
    """
    if threshold_pp <= 0.0:
        return []
    target_by_id = {m.market_id: m for m in targets}
    out: list[str] = []
    for market_id, prob in probabilities.items():
        market = target_by_id.get(market_id)
        if market is None:
            continue
        bid = market.quote.best_bid
        ask = market.quote.best_ask
        if bid is None or ask is None:
            continue
        try:
            mkt_mid = (float(bid) + float(ask)) / 2.0
            p_mean = float(prob.p_mean)
        except (TypeError, ValueError):
            continue
        edge_pp = abs(p_mean - mkt_mid) * 100.0
        if edge_pp >= float(threshold_pp):
            out.append(market_id)
    return out


def _opus_candidate_market_ids(
    *,
    targets: list[MarketView],
    selection: SelectionPassResult | None,
    probabilities: dict[str, MarketProbability],
    opus_config: OpusEscalationConfig,
) -> list[str]:
    """Rank Opus candidates by (priority, market_id).

    Priority order (smaller = higher priority):

    1. Exploration-promoted markets (when ``for_exploration=True``).
    2. Markets with a Sonnet probability that meaningfully diverges from
       the quote mid (``|p_sonnet - mid| >= min_sonnet_edge``).
    3. If ``direct_when_tiny_candidate_set`` and ``len(targets) <= 2``,
       the top-1 selected market falls through here even without a Sonnet
       result (Sonnet may have errored).
    """
    exploration_ids: set[str] = set()
    if selection is not None and opus_config.for_exploration:
        exploration_ids = set(selection.exploration_promotion_ids)
    candidates: list[tuple[int, str]] = []
    for market in targets:
        mid_market = compute_market_prior(
            best_bid=market.quote.best_bid, best_ask=market.quote.best_ask,
        )
        sonnet_prob = probabilities.get(market.market_id)
        if market.market_id in exploration_ids:
            candidates.append((0, market.market_id))
            continue
        if (
            sonnet_prob is not None
            and mid_market is not None
            and abs(float(sonnet_prob.p_mean) - float(mid_market)) >= opus_config.min_sonnet_edge
        ):
            candidates.append((1, market.market_id))
            continue
        if opus_config.direct_when_tiny_candidate_set and len(targets) <= 2:
            candidates.append((2, market.market_id))
    candidates.sort()
    return [cid for _prio, cid in candidates]


def _combine_sonnet_opus(
    *,
    p_sonnet: float,
    sigma_sonnet: float,
    p_opus: float,
    sigma_opus: float,
    disagreement_threshold: float,
) -> tuple[float, float, str]:
    """Inverse-variance blend with a disagreement gate.

    Returns ``(p_combined, sigma_combined, label)`` where ``label`` is one
    of ``"agree"`` or ``"disagree"``. When ``|p_o - p_s| >= threshold`` we
    still return a probability (callers may want the blended value for
    diagnostics) but the strategy treats it as a hold by inspecting the
    audit-row ``edge_source``.
    """
    diff = abs(float(p_opus) - float(p_sonnet))
    p_blend, sigma_blend = inverse_variance_blend(p_sonnet, sigma_sonnet, p_opus, sigma_opus)
    if diff >= disagreement_threshold:
        return p_blend, sigma_blend, "disagree"
    return p_blend, sigma_blend, "agree"


def _run_opus_escalation(
    *,
    ctx: TickContext,
    targets: list[MarketView],
    selection: SelectionPassResult | None,
    probabilities: dict[str, MarketProbability],
    batch: ForecastBatchResult,
    audit_rows: list[dict[str, Any]],
    notes: list[str],
    opus_provider: ForecastProvider,
    opus_config: OpusEscalationConfig,
    spend_governor: SpendGovernor | None,
    forecast_cfg: ForecastConfig,
    calibrator: Calibrator | None,
    cost_estimator: Callable[[str], float] | None,
    cache_ttl_sec: int,
    cache: ForecastCache | None,
    monotonic: Callable[[], float],
    deadline_monotonic: float | None,
    per_call_buffer_sec: float,
    qh_features: dict[str, Any] | None,
    # Phase 6C repair: the runner's authoritative ``web_search_enabled``
    # decision (post-gating) is threaded through so Opus respects the
    # same blocks (thin universe / spend cap / deadline) the Sonnet path
    # honored. Plus the category-stamped request overrides built once
    # per pass and reused across both providers.
    web_search_enabled: bool = True,
    request_overrides: dict[str, ForecastRequest] | None = None,
    # Phase 6E auto-escalation: when set, restrict the Opus candidate
    # set to this list of market_ids. The standard candidate-selection
    # logic (exploration / disagreement / direct) is bypassed; the
    # function processes ``restrict_to_market_ids`` in the same order
    # the caller provided, capped by ``opus_config.max_calls_per_tick``.
    # The ``auto_escalated`` flag is stamped into each audit row so
    # reports can separate "auto" from "by-config" Opus calls.
    restrict_to_market_ids: list[str] | None = None,
    auto_escalated: bool = False,
) -> None:
    """Re-forecast a small set of markets with Opus and combine.

    Mutates ``probabilities``, ``batch``, ``audit_rows``, and ``notes`` in
    place. Honours the per-tick call cap, the spend governor, and the
    per-call deadline buffer. Errors never bubble out -- a failure is
    recorded as an audit row with ``edge_source="forecast_error"``.
    """
    target_by_id = {m.market_id: m for m in targets}
    if restrict_to_market_ids is not None:
        # Phase 6E auto-escalation: caller pre-selected the candidates.
        # Preserve order; drop unknown ids.
        candidate_ids = [
            mid for mid in restrict_to_market_ids if mid in target_by_id
        ]
    else:
        candidate_ids = _opus_candidate_market_ids(
            targets=targets,
            selection=selection,
            probabilities=probabilities,
            opus_config=opus_config,
        )
    if not candidate_ids:
        return
    opus_model = getattr(opus_provider, "model", "openrouter:anthropic/claude-opus-4")
    if cost_estimator is not None:
        est_cost = cost_estimator(opus_model)
    else:
        est_cost = estimate_cost_usd(opus_model, config=forecast_cfg)
    sigma_opus_tier = clip_sigma_p(forecast_cfg.sigma_for_tier("opus"))
    sigma_sonnet_tier = clip_sigma_p(forecast_cfg.sigma_for_tier("sonnet"))
    calls_made = 0
    for market_id in candidate_ids:
        if calls_made >= opus_config.max_calls_per_tick:
            break
        market = target_by_id.get(market_id)
        if market is None:
            continue
        # Per-call deadline gate.
        if (
            deadline_monotonic is not None
            and (deadline_monotonic - monotonic()) < per_call_buffer_sec
        ):
            audit_rows.append(forecast_audit_row(
                market_id=market_id,
                ctx=ctx,
                edge_source="forecast_blocked_deadline",
                decision="skip",
                error="deadline_near",
                model=opus_model,
                model_tier="opus",
            ))
            notes.append(f"opus_skipped_deadline:{market_id}")
            continue
        # Spend gate.
        if spend_governor is not None:
            spend_decision = spend_governor.can_call("opus", est_cost)
            if not spend_decision.allowed:
                audit_rows.append(forecast_audit_row(
                    market_id=market_id,
                    ctx=ctx,
                    edge_source="forecast_blocked_spend",
                    decision="skip",
                    error=spend_decision.reason,
                    model=opus_model,
                    model_tier="opus",
                ))
                notes.append(f"opus_blocked_spend:{market_id}")
                continue
        # Issue the Opus call. Reuse the category-stamped request the
        # runner already built for Sonnet so Opus sees the same playbook.
        request = (
            (request_overrides or {}).get(market.market_id)
            or _build_request(market)
        )
        try:
            try:
                result = opus_provider.forecast_one(
                    request, web_search_enabled_override=web_search_enabled,
                )
            except TypeError:
                result = opus_provider.forecast_one(request)
        except Exception as exc:  # provider should never raise but be safe
            result = ForecastResult(
                market_id=market_id,
                error=f"opus_exception: {type(exc).__name__}",
                edge_source="forecast_error",
                model=opus_model,
                model_tier="opus",
            )
        calls_made += 1
        if result.error is not None or result.p_raw is None:
            audit_rows.append(forecast_audit_row(
                market_id=market_id,
                ctx=ctx,
                edge_source=result.edge_source or "forecast_error",
                decision="error",
                error=result.error or "unknown_error",
                api_cost_usd=result.api_cost_usd,
                model=result.model or opus_model,
                model_tier="opus",
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cached_tokens=result.cached_tokens,
            ))
            continue
        # Calibrate + clip sigma.
        result.model_tier = "opus"
        result.sigma_p = _resolve_sigma(
            explicit_sigma=result.sigma_p,
            model_tier="opus",
            forecast_config=forecast_cfg,
        )
        p_cal_opus = calibrate_probability(result.p_raw, calibrator or Calibrator(method="identity"))
        result.p_mean = p_cal_opus
        # Phase 6C audit polish: stamp category from the runner-side
        # request stamp and compute evidence_hash so the Opus
        # forecasts.audit_json carries the same Phase 6B fields the
        # Sonnet path emits. Pure audit -- no trading-behavior change.
        if request.category and not result.category:
            result.category = request.category
        if web_search_enabled:
            result.evidence_hash = compute_evidence_hash(
                cited_urls=result.cited_urls,
                web_search_requests=result.web_search_requests,
            )
        if spend_governor is not None and result.api_cost_usd > 0:
            try:
                spend_governor.record(result.api_cost_usd)
            except ValueError:
                pass
        batch.attempted += 1
        batch.succeeded += 1
        batch.spend_estimate_usd += result.api_cost_usd
        batch.results.append(result)

        sonnet_prob = probabilities.get(market_id)
        # Phase 6R: correct combine order. Combine raw calibrated model
        # probabilities FIRST, then apply the market-prior blend ONCE to
        # the combined model estimate. The prior version blended
        # Sonnet+market and then combined that blended value with Opus,
        # which double-counted the market prior.
        if sonnet_prob is not None and sonnet_prob.p_model_only is not None:
            p_sonnet_model = float(sonnet_prob.p_model_only)
            sigma_sonnet_model = float(sonnet_prob.sigma_model_only or sigma_sonnet_tier)
            p_combined_model, sigma_combined_model, label = _combine_sonnet_opus(
                p_sonnet=p_sonnet_model,
                sigma_sonnet=sigma_sonnet_model,
                p_opus=float(p_cal_opus),
                sigma_opus=float(result.sigma_p or sigma_opus_tier),
                disagreement_threshold=opus_config.disagreement_threshold,
            )
            # Apply the market-prior blend ONCE to the combined model est.
            blend = _blend_for_market(
                market=market,
                p_model_cal=p_combined_model,
                sigma_model=sigma_combined_model,
                ctx=ctx,
                optional_extra_shrinkage=_qh_shrinkage(qh_features, market_id),
            )
            if label == "disagree":
                edge_source = EDGE_MODEL_DISAGREEMENT
            elif market_id in (selection.exploration_promotion_ids if selection else []):
                edge_source = EDGE_OPUS_EXPLORATION
            else:
                edge_source = EDGE_SONNET_OPUS_CONFIRMED
            probabilities[market_id] = MarketProbability(
                p_mean=blend.p_blend,
                sigma_p=blend.sigma_blend,
                edge_source=edge_source,
                model_tier="opus",
                p_model_only=p_combined_model,
                sigma_model_only=sigma_combined_model,
            )
            audit_extra = {
                "opus_pair_label": label,
                "p_sonnet_model": p_sonnet_model,
                "sigma_sonnet_model": sigma_sonnet_model,
                "p_opus_model": float(p_cal_opus),
                "sigma_opus_model": float(result.sigma_p or sigma_opus_tier),
                "p_model_combined": float(p_combined_model),
                "sigma_model_combined": float(sigma_combined_model),
                "p_market": float(blend.p_market),
                "p_blend_final": float(blend.p_blend),
                "sigma_blend_final": float(blend.sigma_blend),
                "blend_reason": blend.blend_reason,
            }
        elif sonnet_prob is not None:
            # Fallback: no pre-blend model probability available (Phase 6
            # legacy path). Fall back to the prior combine to keep
            # backwards compatibility but flag the audit.
            p_combined, sigma_combined, label = _combine_sonnet_opus(
                p_sonnet=float(sonnet_prob.p_mean),
                sigma_sonnet=float(sonnet_prob.sigma_p or sigma_sonnet_tier),
                p_opus=float(p_cal_opus),
                sigma_opus=float(result.sigma_p or sigma_opus_tier),
                disagreement_threshold=opus_config.disagreement_threshold,
            )
            if label == "disagree":
                edge_source = EDGE_MODEL_DISAGREEMENT
            elif market_id in (selection.exploration_promotion_ids if selection else []):
                edge_source = EDGE_OPUS_EXPLORATION
            else:
                edge_source = EDGE_SONNET_OPUS_CONFIRMED
            probabilities[market_id] = MarketProbability(
                p_mean=p_combined,
                sigma_p=sigma_combined,
                edge_source=edge_source,
                model_tier="opus",
            )
            audit_extra = {
                "opus_pair_label": label,
                "p_sonnet_pre_combine": float(sonnet_prob.p_mean),
                "sigma_sonnet_pre_combine": float(sonnet_prob.sigma_p),
                "phase6r_warning": "legacy_combine_path_missing_p_model_only",
            }
        else:
            # No Sonnet result: opus is the only signal. Blend with market.
            blend = _blend_for_market(
                market=market,
                p_model_cal=p_cal_opus,
                sigma_model=result.sigma_p,
                ctx=ctx,
                optional_extra_shrinkage=_qh_shrinkage(qh_features, market_id),
            )
            label = "agree"
            if (
                selection is not None
                and market_id in selection.exploration_promotion_ids
            ):
                edge_source = EDGE_OPUS_EXPLORATION
            else:
                edge_source = EDGE_OPUS_DIRECT
            probabilities[market_id] = MarketProbability(
                p_mean=blend.p_blend,
                sigma_p=blend.sigma_blend,
                edge_source=edge_source,
                model_tier="opus",
                p_model_only=float(p_cal_opus),
                sigma_model_only=float(result.sigma_p or sigma_opus_tier),
            )
            audit_extra = {
                "opus_pair_label": label,
                "p_opus_model": float(p_cal_opus),
                "sigma_opus_model": float(result.sigma_p or sigma_opus_tier),
                "p_market": float(blend.p_market),
                "p_blend_final": float(blend.p_blend),
                "sigma_blend_final": float(blend.sigma_blend),
                "blend_reason": blend.blend_reason,
            }

        # Phase 6C audit polish: when Opus actually retrieved web
        # evidence, promote the edge_source label so funnel + evidence
        # reports surface "web_opus_forecast" instead of the bare combine
        # outcome -- but preserve `model_disagreement` (trading layer
        # gates on this) and `opus_exploration` (carries strategy
        # semantics: this market entered the forecast pass via the
        # exploration selector, not via Opus's web evidence). The
        # combine label itself is queryable through audit_json.opus_pair_label
        # so no analytics granularity is lost.
        cited_urls_list = list(result.cited_urls or ())
        if (
            web_search_enabled
            and cited_urls_list
            and edge_source not in (EDGE_MODEL_DISAGREEMENT, EDGE_OPUS_EXPLORATION)
        ):
            edge_source = EDGE_WEB_OPUS_FORECAST
            result.edge_source = EDGE_WEB_OPUS_FORECAST
            prev = probabilities.get(market_id)
            if prev is not None:
                probabilities[market_id] = MarketProbability(
                    p_mean=prev.p_mean,
                    sigma_p=prev.sigma_p,
                    edge_source=EDGE_WEB_OPUS_FORECAST,
                    model_tier=prev.model_tier,
                    p_model_only=prev.p_model_only,
                    sigma_model_only=prev.sigma_model_only,
                )
        # Phase 6C audit polish: always merge Phase 6B evidence fields
        # into audit_extra so the Opus row matches the Sonnet row shape,
        # even when no citations were retrieved (then cited_urls=[] and
        # web_search_requests=0). Without this, the evidence_report's
        # forecasts_with_evidence counter under-reports Opus rows.
        audit_extra.update({
            "category": result.category,
            "evidence_quality": result.evidence_quality,
            "evidence_hash": result.evidence_hash,
            "cited_urls": cited_urls_list,
            "key_drivers_json": result.key_drivers_json,
            "stale_evidence": result.stale_evidence,
            "web_search_requests": int(result.web_search_requests or 0),
            "web_search_enabled": bool(web_search_enabled),
            # Phase 6E: distinguish auto-escalated Opus calls (the
            # budget profile would have skipped Opus, but the Sonnet
            # blend produced an actionable edge) from by-config calls.
            "opus_auto_escalated": bool(auto_escalated),
        })

        audit_rows.append(forecast_audit_row(
            market_id=market_id,
            ctx=ctx,
            edge_source=edge_source,
            decision="ok",
            p_raw=result.p_raw,
            p_cal=p_cal_opus,
            sigma_p=result.sigma_p,
            rationale=result.rationale,
            api_cost_usd=result.api_cost_usd,
            model=result.model or opus_model,
            model_tier="opus",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cached_tokens=result.cached_tokens,
            extra=audit_extra,
        ))
        if label == "disagree":
            notes.append(f"opus_disagreement:{market_id}")


def _tier_for_governor(provider: ForecastProvider, market: MarketView | None = None) -> str:
    """Map a provider's model_tier into a governor-friendly key."""
    cfg = getattr(provider, "config", None)
    if cfg is not None:
        return getattr(cfg, "model_tier", "sonnet")
    return "sonnet"
