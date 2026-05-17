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
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable

from kalibre.calibrate import Calibrator, calibrate_probability
from kalibre.forecast.blend import (
    BlendResult,
    blend_with_market_prior,
    compute_market_prior,
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
from kalibre.layers import LayersConfig
from kalibre.selection import (
    SelectionPassResult,
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

    @property
    def summary(self) -> dict[str, Any]:
        base = self.batch.summary()
        if self.selection is not None:
            sel_summary = self.selection.summary
            base["selected_count"] = sel_summary["selected_count"]
            base["selected_market_ids"] = sel_summary["selected_market_ids"]
            base["selection_skip_reason_counts"] = sel_summary["skip_reason_counts"]
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
    model = getattr(provider, "model", "unknown")
    provider_tier = _tier_for_governor(provider, None)
    accepted, _decisions = filter_universe(ctx.markets, now=ctx.now, config=universe_config)
    # Phase 4B: deterministic target selector. ``passthrough_selector`` is
    # used as the safe fallback when callers explicitly opt out (legacy
    # tests). Production callers pass ``selector=select_forecast_targets``
    # so the Sonnet budget lands on tradable edges, not the first N.
    selector_fn = selector or passthrough_selector
    selection = selector_fn(
        ctx=ctx,
        markets=accepted,
        max_markets=int(max_markets),
        portfolio=ctx.portfolio,
        qh_features=qh_features,
        layer_config=layers_config,
        degraded_mode=degraded_mode,
    )
    targets = list(selection.selected)

    batch = ForecastBatchResult(model=model)
    audit_rows: list[dict[str, Any]] = []
    probabilities: dict[str, MarketProbability] = {}
    notes: list[str] = []

    if not targets:
        notes.append("no_universe_candidates_for_forecast")
        return ForecastPassResult(
            batch=batch, probabilities=probabilities, audit_rows=audit_rows, notes=notes,
            selection=selection,
        )

    # Deadline pre-check: skip every call if we don't have headroom.
    if (
        deadline_monotonic is not None
        and (deadline_monotonic - monotonic()) < deadline_skip_buffer_sec
    ):
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
        request = _build_request(market)
        key = cache_key(request, model)
        cached = cache.get(key, ttl_sec=cache_ttl_sec)
        if cached is not None:
            batch.cached += 1
            batch.succeeded += 1
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

        # Issue the call.
        result = provider.forecast_one(request)
        if result.error is not None or result.p_raw is None:
            batch.failed += 1
            batch.results.append(result)
            audit_rows.append(forecast_audit_row(
                market_id=market.market_id,
                ctx=ctx,
                edge_source=result.edge_source or "forecast_error",
                decision="error",
                error=result.error or "unknown_error",
                api_cost_usd=result.api_cost_usd,
                model=result.model or model,
                model_tier=result.model_tier or provider_tier,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cached_tokens=result.cached_tokens,
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
        # Cache stores the raw post-provider result (without cache_hit flip).
        # The blend is recomputed every tick from the current quote, so we
        # deliberately do NOT cache the blended probability.
        cache.put(key, result)

        if spend_governor is not None and result.api_cost_usd > 0:
            try:
                spend_governor.record(result.api_cost_usd)
            except ValueError:
                pass
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
        ))

    if isinstance(provider, NoOpForecastProvider):
        notes.append("provider:noop")

    return ForecastPassResult(
        batch=batch, probabilities=probabilities, audit_rows=audit_rows, notes=notes,
        selection=selection,
    )


def _tier_for_governor(provider: ForecastProvider, market: MarketView | None = None) -> str:
    """Map a provider's model_tier into a governor-friendly key."""
    cfg = getattr(provider, "config", None)
    if cfg is not None:
        return getattr(cfg, "model_tier", "sonnet")
    return "sonnet"
