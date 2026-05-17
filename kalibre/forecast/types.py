"""Typed forecast inputs/outputs.

These are deliberately plain dataclasses with optional fields so we can
serialize them to the audit log without losing information when the
provider errors out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class ForecastRequest:
    """Inputs the prompt builder needs to produce a calibrated forecast.

    All fields are taken from the SDK ``MarketData`` shape (already
    normalized through :class:`kalibre.tick_context.MarketView`).
    """

    market_id: str
    question: str
    source: str | None
    topic: str | None
    family: str | None
    resolution_time: datetime | None
    best_bid: float
    best_ask: float
    spread: float | None
    volume_24h: float | None
    rules: str | None = None
    description: str | None = None
    # Phase 6B: deterministic category stamp (e.g. "soccer", "nfl",
    # "weather"). Computed by the runner via kalibre.categories.
    category: str | None = None

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0


@dataclass
class ForecastResult:
    """Per-market forecast outcome (success or failure).

    ``sigma_p`` defaults to ``None`` to mean "no explicit sigma".
    Downstream code (the runner) resolves this against the A.3 tier
    sigma table so the Sonnet/Opus/Haiku/structural defaults actually
    apply. A provider may set ``sigma_p`` explicitly when the model
    reports its own confidence; the runner clips that value into
    ``[0.02, 0.25]`` but otherwise preserves it.
    """

    market_id: str
    p_raw: float | None = None
    p_mean: float | None = None  # post-calibration; equals p_raw until calibrated
    sigma_p: float | None = None
    confidence: float | None = None
    model_tier: str = "external"
    edge_source: str = "sonnet_forecast"
    rationale: str = ""
    api_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_hit: bool = False
    error: str | None = None
    model: str = ""
    # Phase 6B: evidence-aware forecast extras. All additive, defaulted so
    # pre-6B callers / cache rows stay valid.
    category: str | None = None
    evidence_quality: float | None = None
    evidence_hash: str | None = None
    source_ids: tuple[str, ...] = field(default_factory=tuple)
    cited_urls: tuple[str, ...] = field(default_factory=tuple)
    key_drivers_json: str | None = None
    stale_evidence: bool | None = None
    web_search_requests: int = 0

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.p_raw is not None


@dataclass
class ForecastBatchResult:
    """Aggregate over one tick's forecast pass."""

    results: list[ForecastResult] = field(default_factory=list)
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    cached: int = 0
    spend_estimate_usd: float = 0.0
    model: str = ""
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "cached": self.cached,
            "spend_estimate_usd": round(self.spend_estimate_usd, 6),
            "model": self.model,
        }
