"""Kalibre T0 survival shell.

CLI entrypoint: ``python -m kalibre.loop``

The loop:

1. Acquires a single-writer PID file and starts a heartbeat thread.
2. Creates or resumes the experiment + participant.
3. Repeats until ``max_ticks`` is reached or the server reports
   ``experiment_completed``:
   - claim a tick (with exp-backoff),
   - load candidates, fetch portfolio (best effort),
   - run the T0 strategy (no-op),
   - put_plan, submit_intents (always — even an empty list),
   - finalize, complete_tick,
   - persist tick + intent audit rows to SQLite,
   - append a row to summary.csv and refresh live_progress.md.

Failures inside the tick body NEVER skip finalize+complete. Every call
that touches the server is wrapped in :func:`with_backoff` so transient
errors don't kill the loop.

T0 deliberately submits no trades. Phase 2 plugs a real strategy into
:meth:`KalibreLoop._strategy_t0`.

Environment:

- ``PA_SERVER_URL`` (default: SDK ``DEFAULT_API_URL``)
- ``PA_SERVER_API_KEY`` (required)
- ``PA_EXPERIMENT_SLUG`` (default: timestamped slug)
- ``PA_MAX_TICKS`` (default: 1)
- ``PA_STARTING_CASH`` (default: 10000)
- ``KALIBRE_EXPERIMENTS_DIR`` (default: ``experiments``)
- ``KALIBRE_PID_DIR`` (default: ``.kalibre/pids``)
- ``KALIBRE_HEARTBEAT_INTERVAL_SEC`` (default: 60)
- ``KALIBRE_SUBMIT_CUTOFF_SEC`` (default: 480)
- ``KALIBRE_CLAIM_IDLE_SLEEP_SEC`` (default: 15)
- ``KALIBRE_CHECKPOINT_EVERY_N_TICKS`` (default: 4)
- ``KALIBRE_LOG_LEVEL`` (default: INFO)
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, TypeVar

try:  # python-dotenv is optional for environments that already export the vars.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]

from ai_prophet_core import (
    APIError,
    DEFAULT_API_URL,
    ServerAPIClient,
    TradeIntentRequest,
)
from ai_prophet_core.arena import BenchmarkSession, SubmissionResult, TickLease

from kalibre import __version__ as KALIBRE_VERSION
from kalibre.calibrate import Calibrator, load_or_fit_calibrator, save_calibrator
from kalibre.forecast import (
    ForecastCache,
    ForecastPassResult,
    SQLiteForecastCache,
    build_default_provider,
    run_forecast_pass,
)
from kalibre.forecast.provider import ForecastProvider, NoOpForecastProvider
from kalibre.canary import CanaryClipReport, CanaryConfig, apply_canary_clip
from kalibre.layers import LayersConfig, get_default_layers_config
from kalibre.longshot import DECISION_ELIGIBLE as _LS_ELIGIBLE
from kalibre.longshot import LongshotPassResult, run_longshot_pass
from kalibre.progress import ExperimentDir, utcnow_iso
from kalibre.quote_history import (
    QuoteHistoryPassResult,
    build_confidence_shadow_rows,
    build_deference_shadow_rows,
    build_features_shadow_rows,
    run_quote_history_pass,
)
from kalibre.selection import (
    SelectionPassResult,
    passthrough_selector,
    select_forecast_targets,
)
from kalibre.spend import SpendGovernor
from kalibre.state import StateStore
from kalibre.strategy import (
    MarketProbability,
    StrategyEnvConfig,
    StrategyMode,
    StrategyResult,
    deterministic_dry_run_strategy,
    parse_strategy_env,
    t0_noop_strategy,
)
from kalibre.tick_context import build_tick_context
from kalibre.watchdog import (
    AlreadyRunningError,
    Heartbeat,
    ensure_single_writer,
    release_pid_file,
)


EDGE_SOURCE_NOOP = "t0_noop"
STRATEGY_NAME = "kalibre-t0-survival"

# Provider type for late-bound probability lookup (T1 will wire the LLM here).
ProbabilitiesProvider = Callable[
    ["TickContext"], dict[str, MarketProbability]  # noqa: F821 - forward ref
]

T = TypeVar("T")
logger = logging.getLogger("kalibre.loop")


# --- backoff -----------------------------------------------------------------


class GiveUp(Exception):
    """Raised when a retried operation runs out of budget."""


def with_backoff(
    fn: Callable[[], T],
    *,
    name: str,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    deadline_monotonic: float | None = None,
    retry_on: tuple[type[BaseException], ...] = (APIError, OSError),
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    rng: random.Random | None = None,
) -> T:
    """Exponential backoff with jitter.

    Args:
        fn: Zero-arg callable to invoke.
        name: Short identifier for logging/telemetry.
        max_attempts: Stop after this many tries.
        base_delay/max_delay: Backoff bounds in seconds.
        deadline_monotonic: Absolute monotonic deadline; raises if past.
        retry_on: Exceptions to swallow and retry.
        on_retry: Optional callback ``(attempt, exc, sleep_sec)``.
        sleep/monotonic/rng: Injection seams for tests.
    """
    rng = rng or random
    attempts = 0
    while True:
        attempts += 1
        try:
            return fn()
        except retry_on as exc:
            if deadline_monotonic is not None and monotonic() >= deadline_monotonic:
                raise
            if attempts >= max_attempts:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempts - 1)))
            delay += rng.uniform(0.0, 0.5)
            if deadline_monotonic is not None:
                remaining = deadline_monotonic - monotonic()
                if remaining <= 0:
                    raise
                delay = min(delay, remaining)
            if on_retry is not None:
                with contextlib.suppress(Exception):
                    on_retry(attempts, exc, delay)
            sleep(delay)


# --- config ------------------------------------------------------------------


@dataclass
class LoopConfig:
    slug: str
    max_ticks: int
    starting_cash: float
    submit_cutoff_sec: float
    claim_idle_sleep_sec: float
    experiments_dir: Path
    pid_dir: Path
    heartbeat_interval_sec: float
    checkpoint_every_n_ticks: int

    @classmethod
    def from_env(cls) -> "LoopConfig":
        ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        slug = _normalize_eval_slug(
            os.getenv("PA_EXPERIMENT_SLUG"),
            team_name=os.getenv("KALIBRE_TEAM_NAME"),
            ts=ts,
        )
        return cls(
            slug=slug,
            max_ticks=int(os.getenv("PA_MAX_TICKS", "1500")),
            starting_cash=float(os.getenv("PA_STARTING_CASH", "10000")),
            submit_cutoff_sec=float(os.getenv("KALIBRE_SUBMIT_CUTOFF_SEC", "480")),
            claim_idle_sleep_sec=float(os.getenv("KALIBRE_CLAIM_IDLE_SLEEP_SEC", "15")),
            experiments_dir=Path(os.getenv("KALIBRE_EXPERIMENTS_DIR", "experiments")),
            pid_dir=Path(os.getenv("KALIBRE_PID_DIR", ".kalibre/pids")),
            heartbeat_interval_sec=float(os.getenv("KALIBRE_HEARTBEAT_INTERVAL_SEC", "60")),
            checkpoint_every_n_ticks=int(os.getenv("KALIBRE_CHECKPOINT_EVERY_N_TICKS", "4")),
        )


def _slugify_team_name(raw: str | None) -> str:
    text = (raw or "team").strip().lower()
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in text)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or "team"


def _normalize_eval_slug(raw_slug: str | None, *, team_name: str | None, ts: str) -> str:
    team = _slugify_team_name(team_name)
    prefix = f"eval_{team}"
    if raw_slug:
        slug = raw_slug.strip()
        return slug if slug.startswith("eval_") else f"{prefix}-{slug}"
    return f"{prefix}-{ts}"


# --- helpers ----------------------------------------------------------------


def _build_config_hash(payload: dict[str, Any]) -> str:
    canon = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()[:16]


def _resolve_lease_deadline_monotonic(
    lease: TickLease, submit_cutoff_sec: float,
) -> float:
    """Translate the wall-clock submit cutoff to a monotonic deadline."""
    tick_ts = lease.tick_ts
    if tick_ts is None:
        return time.monotonic() + submit_cutoff_sec
    cutoff_wall = tick_ts + timedelta(seconds=submit_cutoff_sec)
    delta_sec = (cutoff_wall - datetime.now(tz=UTC)).total_seconds()
    return time.monotonic() + max(0.0, delta_sec)


def _short(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text[:240]


def _safe_repr(payload: dict[str, Any]) -> str:
    try:
        return json.dumps(payload, default=str)[:600]
    except Exception:
        return repr(payload)[:600]


# --- session factory --------------------------------------------------------


def default_session_factory(api_url: str, api_key: str) -> tuple[ServerAPIClient, BenchmarkSession]:
    api = ServerAPIClient(base_url=api_url, api_key=api_key, timeout=30)
    return api, BenchmarkSession(api)


def _make_minimal_ctx(loop: "KalibreLoop", lease: TickLease) -> Any:
    """Build a minimal :class:`TickContext` for strategies that don't need
    candidates (T0 noop). Lets us call the strategy uniformly without
    erroring on missing fields."""
    return build_tick_context(
        tick_id=lease.tick_id or "",
        tick_ts=lease.tick_ts or datetime.now(tz=UTC),
        candidate_set_id=lease.candidate_set_id,
        markets=(),
        portfolio=None,
        experiment_id=loop.experiment_id or "",
        participant_idx=loop.participant_idx or 0,
        version=KALIBRE_VERSION,
    )


# --- the loop ---------------------------------------------------------------


@dataclass
class TickOutcome:
    """In-memory view of a processed tick, used for tests and audit."""

    tick_id: str | None
    status: str
    candidates_loaded: int
    intents_submitted: int
    accepted: int
    rejected: int
    fills: int
    error_code: str | None
    error_detail: str | None
    fallback_triggered: bool
    elapsed_sec: float
    finalize_called: bool
    complete_called: bool


class KalibreLoop:
    def __init__(
        self,
        config: LoopConfig,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        session_factory: Callable[[str, str], tuple[ServerAPIClient | None, BenchmarkSession]] | None = None,
        session: BenchmarkSession | None = None,
        spend: SpendGovernor | None = None,
        strategy_env: StrategyEnvConfig | None = None,
        probabilities_provider: Callable[[Any], dict[str, MarketProbability]] | None = None,
        forecast_provider: ForecastProvider | None = None,
        forecast_cache: ForecastCache | None = None,
        forecast_calibrator: Calibrator | None = None,
        forecast_max_markets: int | None = None,
        forecast_cache_ttl_sec: int | None = None,
        forecast_seed_path: str | Path | None = None,
        forecast_calibrator_cache_path: str | Path | None = None,
        layers_config: LayersConfig | None = None,
        forecast_selector: Any | None = None,
        canary_config: CanaryConfig | None = None,
    ) -> None:
        self.config = config
        self.api_url = api_url
        self.api_key = api_key
        self._session_factory = session_factory
        self._preset_session = session
        self.spend = spend or SpendGovernor()
        self.strategy_env = strategy_env or parse_strategy_env(os.environ)
        self.probabilities_provider = probabilities_provider or (lambda _ctx: {})
        self._forecast_provider_override = forecast_provider
        self._forecast_cache_override = forecast_cache
        self._forecast_calibrator_override = forecast_calibrator
        self.forecast_max_markets = int(
            forecast_max_markets if forecast_max_markets is not None
            else os.environ.get("KALIBRE_MAX_FORECAST_MARKETS", "6") or 6
        )
        self.forecast_cache_ttl_sec = int(
            forecast_cache_ttl_sec if forecast_cache_ttl_sec is not None
            else os.environ.get("KALIBRE_FORECAST_CACHE_TTL_SEC", str(6 * 3600))
        )
        self.forecast_seed_path = Path(
            forecast_seed_path
            or os.environ.get(
                "KALIBRE_CALIBRATION_SEED",
                "ai-prophet-datasets/datasets/sample-resolved/releases/v1.0.0/tasks.jsonl",
            )
        )
        self.forecast_calibrator_cache_path: Path | None = None
        if forecast_calibrator_cache_path is not None:
            self.forecast_calibrator_cache_path = Path(forecast_calibrator_cache_path)
        self._forecast_provider: ForecastProvider | None = None
        self._forecast_cache: ForecastCache | None = None
        self._calibrator: Calibrator | None = None
        self.last_forecast_pass: ForecastPassResult | None = None
        self.started_at = datetime.now(tz=UTC)
        self.pid_path = config.pid_dir / f"{config.slug}.pid"
        self.heartbeat_path = config.pid_dir / f"{config.slug}.heartbeat"
        self.heartbeat = Heartbeat(self.heartbeat_path, interval_sec=config.heartbeat_interval_sec)
        self.experiment_dir: ExperimentDir | None = None
        self.state: StateStore | None = None
        self.session: BenchmarkSession | None = None
        self.api: ServerAPIClient | None = None
        self.experiment_id: str | None = None
        self.participant_idx: int | None = None
        self.ticks_completed = 0
        self.last_tick_id: str | None = None
        self.last_status: str | None = None
        self.outcomes: list[TickOutcome] = []
        self._pid_owned = False
        self.last_strategy_result: StrategyResult | None = None
        self.layers_config = layers_config or get_default_layers_config()
        self.last_quote_history_pass: QuoteHistoryPassResult | None = None
        self.last_longshot_pass: LongshotPassResult | None = None
        # Phase 4B: deterministic forecast-target selector and live-canary clip.
        self.forecast_selector = forecast_selector or select_forecast_targets
        self.canary_config = canary_config or CanaryConfig.from_env()
        self.last_canary_report: CanaryClipReport | None = None
        self.last_selected_market_ids: list[str] = []
        self._last_tick_ctx: Any = None

    # --- public api ---------------------------------------------------------

    def setup(self) -> None:
        ensure_single_writer(self.pid_path)
        self._pid_owned = True
        self.heartbeat.start()
        if self._preset_session is not None:
            self.session = self._preset_session
        else:
            factory = self._session_factory or default_session_factory
            if self.api_url is None or self.api_key is None:
                raise RuntimeError("api_url and api_key required when no session is provided")
            api, session = factory(self.api_url, self.api_key)
            self.api = api
            self.session = session
        config_payload = {
            "strategy": STRATEGY_NAME,
            "version": KALIBRE_VERSION,
            "max_ticks": self.config.max_ticks,
            "starting_cash": self.config.starting_cash,
        }
        config_hash = _build_config_hash(config_payload)
        exp = with_backoff(
            lambda: self.session.create_experiment(
                slug=self.config.slug,
                config_hash=config_hash,
                config_json=config_payload,
                n_ticks=self.config.max_ticks,
            ),
            name="create_experiment",
            max_attempts=5,
            on_retry=lambda attempt, exc, sleep_s: self._log(
                "create_experiment_retry",
                attempt=attempt,
                error=_short(exc),
                sleep_sec=round(sleep_s, 3),
            ),
        )
        self.experiment_id = exp.experiment_id
        self.experiment_dir = ExperimentDir(
            self.config.experiments_dir, self.config.slug, self.experiment_id,
        )
        self.state = StateStore(
            self.experiment_dir.state_db_path,
            checkpoint_dir=self.experiment_dir.checkpoints_dir,
        )
        self.state.connect()
        participant = with_backoff(
            lambda: self.session.upsert_participant(
                model="custom:kalibre-t0",
                rep=0,
                starting_cash=self.config.starting_cash,
            ),
            name="upsert_participant",
            max_attempts=5,
            on_retry=lambda attempt, exc, sleep_s: self._log(
                "upsert_participant_retry",
                attempt=attempt,
                error=_short(exc),
                sleep_sec=round(sleep_s, 3),
            ),
        )
        self.participant_idx = participant.participant_idx
        self._log(
            "experiment_ready",
            experiment_id=self.experiment_id,
            slug=self.config.slug,
            participant_idx=self.participant_idx,
            version=KALIBRE_VERSION,
            config_hash=config_hash,
            target_ticks=self.config.max_ticks,
            submit_cutoff_sec=self.config.submit_cutoff_sec,
        )
        self.experiment_dir.update_live_progress(
            completed=0,
            target=self.config.max_ticks,
            started_at=self.started_at,
            last_tick_id=None,
            last_status="initialized",
        )
        if self.strategy_env.mode == StrategyMode.FORECAST_DRY_RUN:
            self._initialise_forecast_stack()

    def _initialise_forecast_stack(self) -> None:
        """Lazy-init the forecast provider / cache / calibrator for forecast_dry_run."""
        if self._forecast_provider is None:
            self._forecast_provider = self._forecast_provider_override or build_default_provider()
        if self._forecast_cache is None and self.state is not None:
            self._forecast_cache = self._forecast_cache_override or SQLiteForecastCache(self.state.connect())
        if self._calibrator is None:
            if self._forecast_calibrator_override is not None:
                self._calibrator = self._forecast_calibrator_override
            else:
                cache_path = (
                    self.forecast_calibrator_cache_path
                    or (self.experiment_dir.dir / "calibrator.json" if self.experiment_dir else None)
                )
                if cache_path is None:
                    from kalibre.calibrate import fit_seed_calibrator
                    self._calibrator = fit_seed_calibrator(self.forecast_seed_path)
                else:
                    self._calibrator = load_or_fit_calibrator(
                        cache_path=cache_path, seed_path=self.forecast_seed_path,
                    )
                    try:
                        save_calibrator(cache_path, self._calibrator)
                    except OSError:
                        pass
        if self._calibrator is not None and self.state is not None:
            with contextlib.suppress(Exception):
                self.state.upsert_calibrator(
                    bucket_key="global",
                    method=self._calibrator.method,
                    params_json=self._calibrator.to_json(),
                    n_samples=self._calibrator.n_samples,
                    brier=self._calibrator.brier_score,
                    log_loss=self._calibrator.log_loss,
                )
        self._log(
            "forecast_stack_ready",
            model=getattr(self._forecast_provider, "model", "unknown"),
            calibrator_method=self._calibrator.method if self._calibrator else "identity",
            calibrator_n=self._calibrator.n_samples if self._calibrator else 0,
            cache=type(self._forecast_cache).__name__ if self._forecast_cache else "none",
            max_markets=self.forecast_max_markets,
        )

    def run(self) -> int:
        self.setup()
        try:
            while self.ticks_completed < self.config.max_ticks:
                if not self._run_one_tick():
                    break
            return 0
        finally:
            self._shutdown()

    # --- tick body ----------------------------------------------------------

    def _run_one_tick(self) -> bool:
        assert self.session is not None
        assert self.experiment_dir is not None
        assert self.state is not None
        tick_started_monotonic = time.monotonic()
        lease = self._safe_claim_tick()
        if lease is None:
            return True
        if not lease.available:
            if lease.reason == "experiment_completed":
                self._log("experiment_completed")
                return False
            wait = max(1, int(lease.retry_after_sec or self.config.claim_idle_sleep_sec))
            self._log("no_tick_available", reason=lease.reason, retry_after_sec=wait)
            time.sleep(wait)
            return True

        deadline_monotonic = _resolve_lease_deadline_monotonic(
            lease, self.config.submit_cutoff_sec,
        )
        tick_id = lease.tick_id
        seconds_until_cutoff = max(0.0, deadline_monotonic - time.monotonic())
        self._log(
            "tick_claimed",
            tick_id=tick_id,
            candidate_set_id=lease.candidate_set_id,
            submit_cutoff_sec=self.config.submit_cutoff_sec,
            seconds_until_cutoff=round(seconds_until_cutoff, 3),
        )

        # Accumulators for the audit row.
        candidates_loaded = 0
        accepted = 0
        rejected = 0
        fills = 0
        intents: list[TradeIntentRequest] = []
        intents_submitted = 0
        cash = ""
        equity = ""
        total_pnl = ""
        error_code: str | None = None
        error_detail: str | None = None
        fallback_triggered = False
        status = "COMPLETED"
        finalize_called = False
        complete_called = False
        tick = None
        portfolio: Any = None
        strategy_result: StrategyResult | None = None

        skip_strategy = seconds_until_cutoff <= 10.0
        if skip_strategy:
            fallback_triggered = True
            self._log(
                "deadline_skip_strategy",
                tick_id=tick_id,
                seconds_until_cutoff=round(seconds_until_cutoff, 3),
            )
        else:
            try:
                tick = with_backoff(
                    lambda: self.session.load_candidates(lease),
                    name="load_candidates",
                    deadline_monotonic=deadline_monotonic,
                    on_retry=lambda attempt, exc, sleep_s: self._log(
                        "load_candidates_retry",
                        tick_id=tick_id,
                        attempt=attempt,
                        error=_short(exc),
                        sleep_sec=round(sleep_s, 3),
                    ),
                )
                lease = tick.lease
                candidates_loaded = len(tick.candidates.markets)
                self._log(
                    "candidates_loaded",
                    tick_id=tick_id,
                    candidate_set_id=lease.candidate_set_id,
                    candidates_loaded=candidates_loaded,
                )
            except Exception as exc:
                fallback_triggered = True
                error_code = "load_candidates_failed"
                error_detail = _short(exc)
                self._log("load_candidates_failed", tick_id=tick_id, error=_short(exc))

            try:
                portfolio = with_backoff(
                    lambda: self.session.get_portfolio(self.participant_idx),
                    name="get_portfolio",
                    deadline_monotonic=deadline_monotonic,
                )
            except Exception as exc:
                portfolio = None
                fallback_triggered = True
                self._log("get_portfolio_failed", tick_id=tick_id, error=_short(exc))
            if portfolio is not None:
                cash = str(getattr(portfolio, "cash", "") or "")
                equity = str(getattr(portfolio, "equity", "") or "")
                total_pnl = str(getattr(portfolio, "total_pnl", "") or "")

            try:
                strategy_result = self._dispatch_strategy(
                    lease=lease, tick=tick, portfolio=portfolio,
                    deadline_monotonic=deadline_monotonic,
                )
                intents = list(strategy_result.intents)
                # Phase 4B: canary clip after allocation, before submit.
                # The clip is a no-op when canary mode is off or the intent
                # list is already empty (default path). The dispatcher
                # already built the TickContext; reuse it for the lookup.
                ctx_for_clip = self._last_tick_ctx
                if self.canary_config.enabled and ctx_for_clip is not None:
                    markets_by_id = ctx_for_clip.market_by_id()
                    intents, canary_report = apply_canary_clip(
                        intents,
                        config=self.canary_config,
                        markets_by_id=markets_by_id,
                    )
                    self.last_canary_report = canary_report
                    self._log("canary_clip_applied", **canary_report.summary)
                else:
                    self.last_canary_report = CanaryClipReport(
                        enabled=self.canary_config.enabled,
                        intents_before=len(intents),
                        intents_after=len(intents),
                        removed_by_count=0,
                    )
            except Exception as exc:
                fallback_triggered = True
                error_code = error_code or "strategy_exception"
                error_detail = error_detail or _short(exc)
                self._log("strategy_exception", tick_id=tick_id, error=_short(exc))
                intents = []
                strategy_result = None

        plan_json = {
            "version": KALIBRE_VERSION,
            "strategy": STRATEGY_NAME,
            "tick_id": tick_id,
            "candidate_set_id": lease.candidate_set_id,
            "candidates_loaded": candidates_loaded,
            "intent_count": len(intents),
            "fallback_triggered": fallback_triggered,
            "edge_source": (
                strategy_result.mode if strategy_result is not None else EDGE_SOURCE_NOOP
            ),
            "experiment_id": self.experiment_id,
            "participant_idx": self.participant_idx,
            "submit_cutoff_sec": self.config.submit_cutoff_sec,
            "strategy_mode": self.strategy_env.mode.value,
            "live_trades_enabled": self.strategy_env.enable_live_trades,
        }
        if strategy_result is not None:
            plan_json["proposal_summary"] = strategy_result.proposal_summary
            plan_json["notes"] = list(strategy_result.notes)
        if self.last_selected_market_ids:
            plan_json["selected_forecast_target_ids"] = list(self.last_selected_market_ids)
        if self.last_forecast_pass is not None and self.last_forecast_pass.selection is not None:
            plan_json["selection_skip_reason_counts"] = (
                self.last_forecast_pass.selection.skip_reason_counts
            )
        if self.last_canary_report is not None:
            plan_json["canary"] = self.last_canary_report.summary
        try:
            with_backoff(
                lambda: self.session.put_plan(lease, self.participant_idx, plan_json),
                name="put_plan",
                deadline_monotonic=deadline_monotonic,
                on_retry=lambda attempt, exc, sleep_s: self._log(
                    "put_plan_retry",
                    tick_id=tick_id,
                    attempt=attempt,
                    error=_short(exc),
                    sleep_sec=round(sleep_s, 3),
                ),
            )
        except Exception as exc:
            fallback_triggered = True
            error_code = error_code or "put_plan_failed"
            error_detail = error_detail or _short(exc)
            self._log("put_plan_failed", tick_id=tick_id, error=_short(exc))

        intents_submitted = len(intents)
        submission: SubmissionResult | None = None
        try:
            submission = with_backoff(
                lambda: self.session.submit_intents(
                    lease, self.participant_idx, intents,
                ),
                name="submit_intents",
                deadline_monotonic=deadline_monotonic,
                on_retry=lambda attempt, exc, sleep_s: self._log(
                    "submit_intents_retry",
                    tick_id=tick_id,
                    attempt=attempt,
                    error=_short(exc),
                    sleep_sec=round(sleep_s, 3),
                ),
            )
            accepted = submission.accepted
            rejected = submission.rejected
            fills = len(submission.fills)
            self._log(
                "intents_submitted",
                tick_id=tick_id,
                intents_submitted=intents_submitted,
                accepted=accepted,
                rejected=rejected,
                fills=fills,
            )
        except Exception as exc:
            fallback_triggered = True
            if intents_submitted == 0:
                # Spec: log the rejection but still finalize+complete.
                error_code = error_code or "submit_intents_rejected_empty"
            else:
                error_code = error_code or "submit_intents_failed"
            error_detail = error_detail or _short(exc)
            self._log(
                "submit_intents_failed",
                tick_id=tick_id,
                intents_submitted=intents_submitted,
                error=_short(exc),
            )

        # Finalize is non-negotiable. Always try; on hard failure, mark FAILED.
        finalize_status = "COMPLETED"
        try:
            with_backoff(
                lambda: self.session.finalize(
                    lease,
                    self.participant_idx,
                    status=finalize_status,
                    error_code=error_code,
                    error_detail=error_detail,
                ),
                name="finalize",
                deadline_monotonic=None,  # finalize MUST complete even past cutoff
                max_attempts=6,
                on_retry=lambda attempt, exc, sleep_s: self._log(
                    "finalize_retry",
                    tick_id=tick_id,
                    attempt=attempt,
                    error=_short(exc),
                    sleep_sec=round(sleep_s, 3),
                ),
            )
            finalize_called = True
        except Exception as exc:
            status = "FINALIZE_RETRY_FAILED"
            self._log("finalize_failed", tick_id=tick_id, error=_short(exc))
            try:
                with_backoff(
                    lambda: self.session.finalize(
                        lease,
                        self.participant_idx,
                        status="FAILED",
                        error_code="finalize_failed",
                        error_detail=_short(exc),
                    ),
                    name="finalize_failed_fallback",
                    max_attempts=3,
                )
                status = "FAILED"
                finalize_called = True
            except Exception as exc2:
                self._log(
                    "finalize_fallback_failed", tick_id=tick_id, error=_short(exc2),
                )

        # complete_tick is also non-negotiable.
        try:
            with_backoff(
                lambda: self.session.complete_tick(lease),
                name="complete_tick",
                max_attempts=6,
                on_retry=lambda attempt, exc, sleep_s: self._log(
                    "complete_tick_retry",
                    tick_id=tick_id,
                    attempt=attempt,
                    error=_short(exc),
                    sleep_sec=round(sleep_s, 3),
                ),
            )
            complete_called = True
        except Exception as exc:
            self._log("complete_tick_failed", tick_id=tick_id, error=_short(exc))

        elapsed_sec = time.monotonic() - tick_started_monotonic
        deadline_slip_ms = int(max(0.0, time.monotonic() - deadline_monotonic) * 1000)
        run_ts = utcnow_iso()
        summary_row = {
            "run_ts": run_ts,
            "experiment_id": self.experiment_id,
            "slug": self.config.slug,
            "participant_idx": self.participant_idx,
            "tick_id": tick_id,
            "candidate_set_id": lease.candidate_set_id,
            "status": status,
            "candidates_loaded": candidates_loaded,
            "intents_submitted": intents_submitted,
            "accepted": accepted,
            "rejected": rejected,
            "fills": fills,
            "cash": cash,
            "equity": equity,
            "total_pnl": total_pnl,
            "error_code": error_code or "",
            "error_detail": error_detail or "",
            "elapsed_sec": f"{elapsed_sec:.3f}",
            "deadline_slip_ms": deadline_slip_ms,
            "fallback_triggered": int(fallback_triggered),
        }
        self.experiment_dir.append_summary_row(summary_row)

        # Persist tick + intent audit rows. Audit fields per T0 spec:
        # edge_source, tick_ts, version, experiment_id, participant_idx, market_id.
        with contextlib.suppress(Exception):
            self.state.record_tick({
                "tick_ts": tick_id or "",
                "experiment_id": self.experiment_id,
                "participant_idx": self.participant_idx,
                "candidate_set_id": lease.candidate_set_id,
                "status": status,
                "candidates_loaded": candidates_loaded,
                "intents_submitted": intents_submitted,
                "accepted": accepted,
                "rejected": rejected,
                "fills": fills,
                "cash": cash,
                "equity": equity,
                "total_pnl": total_pnl,
                "error_code": error_code,
                "error_detail": error_detail,
                "elapsed_sec": elapsed_sec,
                "deadline_slip_ms": deadline_slip_ms,
                "fallback_triggered": int(fallback_triggered),
            })
        # Phase 5A: rebuild intent rows with proposal provenance so
        # state.intents reflects the real edge_source/version/audit_json
        # rather than a hard-coded ``t0_noop`` placeholder.
        provenance_by_idx = self._build_intent_provenance(strategy_result, intents)
        intent_rows = self._build_intent_rows(
            intents=intents,
            provenance_by_idx=provenance_by_idx,
            tick_id=tick_id,
            fallback_triggered=fallback_triggered,
            strategy_mode=(
                strategy_result.mode if strategy_result is not None else EDGE_SOURCE_NOOP
            ),
        )
        if intent_rows:
            try:
                self.state.record_intents(intent_rows)
            except Exception as exc:
                self._log("intent_audit_persistence_failed", error=_short(exc), rows=len(intent_rows))

        if strategy_result is not None and strategy_result.audit_rows:
            with contextlib.suppress(Exception):
                self.state.record_proposals(strategy_result.audit_rows)

        # Phase 5A: A.9 fill / rejection telemetry. One row per submitted
        # intent. Persistence failures never block finalize/complete.
        self._persist_fill_rows(
            intents=intents,
            provenance_by_idx=provenance_by_idx,
            submission=submission,
            tick_id=tick_id,
        )

        self.last_tick_id = tick_id
        self.last_status = status
        self.ticks_completed += 1
        self.experiment_dir.update_live_progress(
            completed=self.ticks_completed,
            target=self.config.max_ticks,
            started_at=self.started_at,
            last_tick_id=tick_id,
            last_status=status,
        )

        cp_n = max(1, self.config.checkpoint_every_n_ticks)
        if self.ticks_completed % cp_n == 0:
            try:
                ckpt_path = self.state.checkpoint()
                self._log("state_checkpoint", path=str(ckpt_path))
            except Exception as exc:
                self._log("state_checkpoint_failed", error=_short(exc))

        self._log(
            "tick_completed",
            tick_id=tick_id,
            status=status,
            accepted=accepted,
            rejected=rejected,
            fills=fills,
            intents_submitted=intents_submitted,
            candidates_loaded=candidates_loaded,
            elapsed_sec=round(elapsed_sec, 3),
            deadline_slip_ms=deadline_slip_ms,
            fallback_triggered=int(fallback_triggered),
        )
        self.outcomes.append(
            TickOutcome(
                tick_id=tick_id,
                status=status,
                candidates_loaded=candidates_loaded,
                intents_submitted=intents_submitted,
                accepted=accepted,
                rejected=rejected,
                fills=fills,
                error_code=error_code,
                error_detail=error_detail,
                fallback_triggered=fallback_triggered,
                elapsed_sec=elapsed_sec,
                finalize_called=finalize_called,
                complete_called=complete_called,
            )
        )
        return True

    # --- subordinate helpers -----------------------------------------------

    def _safe_claim_tick(self) -> TickLease | None:
        assert self.session is not None
        try:
            return with_backoff(
                lambda: self.session.claim_tick(),
                name="claim_tick",
                max_attempts=5,
                on_retry=lambda attempt, exc, sleep_s: self._log(
                    "claim_tick_retry",
                    attempt=attempt,
                    error=_short(exc),
                    sleep_sec=round(sleep_s, 3),
                ),
            )
        except Exception as exc:
            self._log("claim_tick_failed", error=_short(exc))
            time.sleep(min(self.config.claim_idle_sleep_sec, 30))
            return None

    def _strategy_t0(self) -> list[TradeIntentRequest]:
        """T0 always returns an empty intent list. Override in tests/T1."""
        return []

    def _dispatch_strategy(
        self,
        *,
        lease: TickLease,
        tick: Any,
        portfolio: Any,
        deadline_monotonic: float | None = None,
    ) -> StrategyResult:
        """Dispatch on ``KALIBRE_STRATEGY_MODE``.

        T0_NOOP preserves the previous behavior exactly: an empty intent
        list and no proposal audit rows. DETERMINISTIC_DRY_RUN runs the
        Phase 2 engine and returns proposal audit rows; intents are only
        non-empty when ``KALIBRE_ENABLE_LIVE_TRADES=1``.
        """
        # Allow subclasses / tests to keep using _strategy_t0 override.
        if self.strategy_env.mode == StrategyMode.T0_NOOP:
            legacy = self._strategy_t0()
            if legacy:
                # Tests still inject an intent list via _strategy_t0; keep that working.
                return StrategyResult(
                    mode=StrategyMode.T0_NOOP.value,
                    proposals=[],
                    intents=list(legacy),
                    universe_decisions=(),
                    audit_rows=[],
                    allocation=None,
                    live_trades_enabled=False,
                    notes=["legacy_strategy_t0_override"],
                )
            return t0_noop_strategy(_make_minimal_ctx(self, lease))
        if self.strategy_env.mode in (StrategyMode.DETERMINISTIC_DRY_RUN, StrategyMode.FORECAST_DRY_RUN):
            if tick is None or portfolio is None:
                return StrategyResult(
                    mode=self.strategy_env.mode.value,
                    proposals=[],
                    intents=[],
                    universe_decisions=(),
                    audit_rows=[],
                    allocation=None,
                    live_trades_enabled=self.strategy_env.enable_live_trades,
                    notes=["missing_candidates_or_portfolio"],
                )
            ctx = build_tick_context(
                tick_id=lease.tick_id or "",
                tick_ts=lease.tick_ts or datetime.now(tz=UTC),
                candidate_set_id=lease.candidate_set_id,
                markets=tick.candidates.markets,
                portfolio=portfolio,
                experiment_id=self.experiment_id or "",
                participant_idx=self.participant_idx or 0,
                version=KALIBRE_VERSION,
                pnl_24h=0.0,
                spend_governor_state={
                    "degraded_mode": self.spend.degraded_mode(),
                    "spent_total_usd": self.spend.spent_total_usd,
                    "spent_day_usd": self.spend.spent_day_usd,
                },
            )
            self._last_tick_ctx = ctx
            # --- Phase 4A: quote-history snapshot + features --------------
            qh_pass = self._run_quote_history_pass(ctx)
            qh_features_by_market = qh_pass.features_by_market if qh_pass else {}
            forecast_audit_rows: list[dict[str, Any]] = []
            forecast_summary: dict[str, Any] | None = None
            if self.strategy_env.mode == StrategyMode.FORECAST_DRY_RUN:
                if self._forecast_provider is None:
                    self._initialise_forecast_stack()
                pass_result = run_forecast_pass(
                    ctx,
                    provider=self._forecast_provider,
                    cache=self._forecast_cache,
                    calibrator=self._calibrator,
                    spend_governor=self.spend,
                    max_markets=self.forecast_max_markets,
                    deadline_monotonic=deadline_monotonic,
                    cache_ttl_sec=self.forecast_cache_ttl_sec,
                    qh_features=qh_features_by_market or None,
                    selector=self.forecast_selector,
                    layers_config=self.layers_config,
                    degraded_mode=self.spend.degraded_mode(),
                )
                self.last_forecast_pass = pass_result
                self.last_selected_market_ids = list(
                    pass_result.selection.selected_market_ids
                    if pass_result.selection is not None else []
                )
                forecast_summary = pass_result.summary
                probabilities = pass_result.probabilities
                forecast_audit_rows = pass_result.audit_rows
                self._log(
                    "forecast_pass",
                    summary=pass_result.summary,
                    notes=list(pass_result.notes),
                )
                # Persist selection audit rows as shadow_proposals.
                if pass_result.selection is not None and self.state is not None:
                    selection_rows = pass_result.selection_shadow_rows(ctx=ctx)
                    if selection_rows:
                        try:
                            self.state.record_shadow_proposals(selection_rows)
                        except Exception as exc:
                            self._log(
                                "selection_shadow_persistence_failed",
                                error=_short(exc),
                                rows=len(selection_rows),
                            )
                if forecast_audit_rows and self.state is not None:
                    try:
                        self.state.record_forecasts(forecast_audit_rows)
                    except Exception as exc:
                        self._log(
                            "forecast_audit_persistence_failed",
                            error=_short(exc),
                            rows=len(forecast_audit_rows),
                        )
                    spend_persisted = 0
                    spend_persist_failed = 0
                    for forecast_row in forecast_audit_rows:
                        cost = float(forecast_row.get("api_cost_usd") or 0.0)
                        if cost <= 0:
                            continue
                        if int(forecast_row.get("cache_hit") or 0):
                            # Cache hits are free; never charge spend_log again.
                            continue
                        if forecast_row.get("error"):
                            continue
                        try:
                            self.state.record_spend(
                                model=forecast_row.get("model") or pass_result.batch.model,
                                cost_usd=cost,
                                purpose=forecast_row.get("edge_source") or "forecast",
                            )
                            spend_persisted += 1
                        except Exception as exc:
                            spend_persist_failed += 1
                            self._log(
                                "forecast_spend_persistence_failed",
                                market_id=forecast_row.get("market_id"),
                                cost_usd=cost,
                                error=_short(exc),
                            )
                    if spend_persisted or spend_persist_failed:
                        self._log(
                            "forecast_spend_persisted",
                            persisted=spend_persisted,
                            failed=spend_persist_failed,
                        )
            else:
                probabilities = self.probabilities_provider(ctx) or {}
            # --- Phase 4A repair: QH shadow ablation rows ---------------
            # Features shadow runs every tick; deference + confidence
            # shadows attach to forecasted markets. Default weights keep
            # the primary probabilities unchanged.
            self._emit_qh_shadow_rows(
                ctx, forecast_audit_rows=forecast_audit_rows or None,
            )
            # --- Phase 4A: longshot shadow proposals --------------------
            self._run_longshot_shadow_pass(ctx)
            # Optional primary fallback fill (default disabled).
            longshot_primary_added = self._apply_longshot_primary(
                ctx=ctx, probabilities=probabilities,
            )
            self.last_strategy_result = deterministic_dry_run_strategy(
                ctx,
                probabilities=probabilities,
                enable_live_trades=self.strategy_env.enable_live_trades,
                spent_today_usd=self.spend.spent_day_usd,
                daily_api_hard_cap_usd=self.spend.daily_hard_usd,
            )
            self.last_strategy_result.mode = self.strategy_env.mode.value
            if forecast_summary is not None:
                self.last_strategy_result.notes.append(
                    f"forecast_summary: {forecast_summary}"
                )
            if self.last_quote_history_pass is not None:
                self.last_strategy_result.notes.append(
                    f"quote_history_summary: {self.last_quote_history_pass.summary}"
                )
            if self.last_longshot_pass is not None:
                self.last_strategy_result.notes.append(
                    f"longshot_summary: {self.last_longshot_pass.summary}"
                )
            if longshot_primary_added:
                self.last_strategy_result.notes.append(
                    f"longshot_primary_added: {longshot_primary_added}"
                )
            return self.last_strategy_result
        # Unknown mode -> fail safe.
        return t0_noop_strategy(_make_minimal_ctx(self, lease))

    def _run_quote_history_pass(self, ctx: Any) -> QuoteHistoryPassResult | None:
        """Snapshot + feature pass over every loaded candidate.

        Persistence is best-effort. Phase 3B / Phase 1 behavior is
        preserved if the QH layer is disabled or any DB error fires.
        """
        layer = self.layers_config.quote_history
        if not layer.enabled or self.state is None:
            self.last_quote_history_pass = None
            return None
        try:
            conn = self.state.connect()
        except Exception as exc:
            self._log("quote_history_connect_failed", error=_short(exc))
            self.last_quote_history_pass = None
            return None
        try:
            qh_pass = run_quote_history_pass(
                conn=conn,
                ctx=ctx,
                markets=ctx.markets,
                layer=layer,
            )
        except Exception as exc:
            self._log("quote_history_compute_failed", error=_short(exc))
            self.last_quote_history_pass = None
            return None
        self.last_quote_history_pass = qh_pass
        if qh_pass.snapshot_rows:
            try:
                self.state.record_quote_history(qh_pass.snapshot_rows)
            except Exception as exc:
                self._log(
                    "quote_history_snapshot_persistence_failed",
                    error=_short(exc),
                    rows=len(qh_pass.snapshot_rows),
                )
        if qh_pass.feature_rows:
            try:
                self.state.record_quote_features(qh_pass.feature_rows)
            except Exception as exc:
                self._log(
                    "quote_history_features_persistence_failed",
                    error=_short(exc),
                    rows=len(qh_pass.feature_rows),
                )
        self._log(
            "quote_history_pass",
            summary=qh_pass.summary,
            qh_weights={
                "confidence_modifier_weight": layer.confidence_modifier_weight,
                "deference_signal_weight": layer.deference_signal_weight,
                "exit_signal_weight": layer.exit_signal_weight,
            },
        )
        return qh_pass

    def _emit_qh_shadow_rows(
        self,
        ctx: Any,
        *,
        forecast_audit_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        """Phase 4A repair: persist QH shadow ablation rows.

        ``quote_history_features`` covers every loaded candidate.
        ``quote_history_deference`` and ``quote_history_confidence``
        cover every forecasted market. None of these influence the
        submitted intents -- they are purely for ablation analysis.
        """
        if self.state is None or self.last_quote_history_pass is None:
            return
        features_by_market = self.last_quote_history_pass.features_by_market
        rows: list[dict[str, Any]] = []
        rows.extend(build_features_shadow_rows(
            ctx=ctx, features_by_market=features_by_market,
        ))
        if forecast_audit_rows:
            markets_by_id = ctx.market_by_id()
            rows.extend(build_deference_shadow_rows(
                ctx=ctx,
                markets_by_id=markets_by_id,
                features_by_market=features_by_market,
                forecast_audit_rows=forecast_audit_rows,
            ))
            rows.extend(build_confidence_shadow_rows(
                ctx=ctx,
                features_by_market=features_by_market,
                forecast_audit_rows=forecast_audit_rows,
            ))
        if not rows:
            return
        try:
            self.state.record_shadow_proposals(rows)
        except Exception as exc:
            self._log(
                "qh_shadow_persistence_failed",
                error=_short(exc),
                rows=len(rows),
            )
        self._log(
            "qh_shadow_pass",
            features=sum(1 for r in rows if r["variant_name"] == "quote_history_features"),
            deference=sum(1 for r in rows if r["variant_name"] == "quote_history_deference"),
            confidence=sum(1 for r in rows if r["variant_name"] == "quote_history_confidence"),
        )

    def _apply_longshot_primary(
        self,
        *,
        ctx: Any,
        probabilities: dict[str, MarketProbability],
    ) -> int:
        """Phase 4A repair: when ``longshot.primary_enabled`` is true, eligible
        longshot rows fill in for markets that have no forecast probability.

        Returns the number of markets the longshot prior added. Live
        trading still requires ``KALIBRE_ENABLE_LIVE_TRADES=1``; this
        function only changes which probabilities reach the strategy.
        """
        if self.last_longshot_pass is None:
            return 0
        layer = self.layers_config.longshot
        if not layer.primary_enabled:
            return 0
        added = 0
        for row in self.last_longshot_pass.shadow_rows:
            if row.decision != _LS_ELIGIBLE:
                continue
            if row.market_id in probabilities:
                continue
            if row.p_prior is None:
                continue
            probabilities[row.market_id] = MarketProbability(
                p_mean=float(row.p_prior),
                sigma_p=0.12,  # structural-only sigma per A.3
                edge_source="longshot_prior",
                model_tier="structural_only",
            )
            added += 1
        if added:
            self._log("longshot_primary_promoted", added=added)
        return added

    def _build_intent_provenance(
        self,
        strategy_result: StrategyResult | None,
        intents: list[TradeIntentRequest],
    ) -> dict[int, Any]:
        """Map each surviving intent (by index) back to its accepted proposal.

        The strategy's allocator emits a list of accepted ``Proposal``
        objects in the same order it built the intents. After the canary
        clip the intent list may be shorter but order is preserved, and
        share counts may change while ``(market_id, action, side)`` stay
        stable. We match on that key, falling back to None when nothing
        plausible is found (e.g. T0 noop legacy override or a strategy
        exception).
        """
        if (
            strategy_result is None
            or strategy_result.allocation is None
            or not intents
        ):
            return {}
        proposals_by_key: dict[tuple[str, str, str], Any] = {}
        for prop in strategy_result.allocation.accepted:
            proposals_by_key[(prop.market_id, prop.action, prop.side)] = prop
        out: dict[int, Any] = {}
        for idx, intent in enumerate(intents):
            key = (intent.market_id, intent.action, intent.side)
            prop = proposals_by_key.get(key)
            if prop is not None:
                out[idx] = prop
        return out

    def _build_intent_rows(
        self,
        *,
        intents: list[TradeIntentRequest],
        provenance_by_idx: dict[int, Any],
        tick_id: str | None,
        fallback_triggered: bool,
        strategy_mode: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, intent in enumerate(intents):
            prop = provenance_by_idx.get(idx)
            if prop is not None:
                edge_source = prop.edge_source
                audit_payload = {
                    "edge_source": prop.edge_source,
                    "tick_ts": tick_id,
                    "version": KALIBRE_VERSION,
                    "experiment_id": self.experiment_id,
                    "participant_idx": self.participant_idx,
                    "market_id": intent.market_id,
                    "side": intent.side,
                    "action": intent.action,
                    "shares": intent.shares,
                    # NOTE: Proposal does not track ``model_tier``; the
                    # edge_source above is the single source of truth
                    # for which forecast layer produced this intent.
                    "p_mean": prop.p_mean,
                    "p_eff": prop.p_eff,
                    "sigma_p": prop.sigma_p,
                    "score": prop.score,
                    "fill_prob": prop.fill_prob,
                    "topic": prop.topic,
                    "source": prop.source,
                    "fallback_triggered": fallback_triggered,
                    "strategy_mode": strategy_mode,
                    "size_usd": prop.size_usd,
                }
            else:
                # No accepted-proposal match -> legacy or unknown
                # path. Fall back to a non-misleading edge_source.
                edge_source = (
                    EDGE_SOURCE_NOOP if strategy_mode == EDGE_SOURCE_NOOP else "unknown"
                )
                audit_payload = {
                    "edge_source": edge_source,
                    "tick_ts": tick_id,
                    "version": KALIBRE_VERSION,
                    "experiment_id": self.experiment_id,
                    "participant_idx": self.participant_idx,
                    "market_id": intent.market_id,
                    "side": intent.side,
                    "action": intent.action,
                    "shares": intent.shares,
                    "fallback_triggered": fallback_triggered,
                    "strategy_mode": strategy_mode,
                }
            rows.append({
                "intent_id": f"{tick_id}:{idx}",
                "tick_ts": tick_id or "",
                "experiment_id": self.experiment_id,
                "participant_idx": self.participant_idx,
                "market_id": intent.market_id,
                "side": intent.side,
                "action": intent.action,
                "shares": intent.shares,
                "edge_source": edge_source,
                "version": KALIBRE_VERSION,
                "audit_json": json.dumps(audit_payload, default=str),
                "fill_status": "",
            })
        return rows

    def _persist_fill_rows(
        self,
        *,
        intents: list[TradeIntentRequest],
        provenance_by_idx: dict[int, Any],
        submission: SubmissionResult | None,
        tick_id: str | None,
    ) -> None:
        """Build and persist one fill audit row per submitted intent.

        Each row maps a submitted intent to whatever the SDK returned
        (FillData / RejectionData) and classifies the outcome as one of
        FILLED, PARTIAL, REJECTED, UNKNOWN. Missing SDK fields stay NULL;
        slippage and post-fill mid are computed only when both sides are
        known.
        """
        if not intents or self.state is None:
            return
        try:
            from ai_prophet_core.client_models import FillData, RejectionData  # noqa: F401
        except Exception:
            FillData = None  # type: ignore[assignment]
            RejectionData = None  # type: ignore[assignment]
        markets_by_id: dict[str, Any] = {}
        if self._last_tick_ctx is not None:
            markets_by_id = self._last_tick_ctx.market_by_id()

        # Match SDK fills + rejections by (market_id, action, side).
        # When multiple fills share a key (multiple intents on the same
        # market+side), use first-match-wins; the API caps to one fill
        # per intent so this is usually unique.
        fills_by_key: dict[tuple[str, str, str], Any] = {}
        rejections_by_key: dict[tuple[str, str, str], Any] = {}
        if submission is not None:
            for fill in submission.fills:
                key = (
                    fill.market_id,
                    str(getattr(fill, "action", "") or ""),
                    str(getattr(fill, "side", "") or ""),
                )
                fills_by_key.setdefault(key, fill)
            for rej in submission.rejections:
                # RejectionData has intent_id + reason; the corresponding
                # intent_id we generate is ``{tick}:{i}`` which generally
                # won't match the server's. Match by intent_id substring
                # if possible, otherwise by index from the rejection's
                # intent_id position.
                rejections_by_key[(str(getattr(rej, "intent_id", "") or ""), "", "")] = rej

        rows: list[dict[str, Any]] = []
        created = datetime.now(tz=UTC).isoformat()
        for idx, intent in enumerate(intents):
            intent_id = f"{tick_id}:{idx}"
            prop = provenance_by_idx.get(idx)
            edge_source = (
                prop.edge_source if prop is not None else "unknown"
            )
            expected_fill_prob = (
                float(prop.fill_prob) if (prop is not None and prop.fill_prob is not None) else None
            )
            market = markets_by_id.get(intent.market_id)
            quoted_bid = quoted_ask = mid = quote_age = submitted_price = None
            if market is not None:
                q = market.quote
                quoted_bid = q.best_bid
                quoted_ask = q.best_ask
                mid = q.mid
                quote_age = q.quote_age_sec(self._last_tick_ctx.now) if self._last_tick_ctx else None
                if intent.side.upper() == "YES" and quoted_ask is not None:
                    submitted_price = float(quoted_ask)
                elif intent.side.upper() == "NO" and quoted_bid is not None:
                    submitted_price = max(1e-9, 1.0 - float(quoted_bid))
            try:
                submitted_shares = float(intent.shares)
            except (TypeError, ValueError):
                submitted_shares = None
            fill = fills_by_key.get((intent.market_id, intent.action, intent.side))
            rejection = None
            # Look up rejection by server intent_id substring (the SDK
            # populates fills[i].intent_id when filled; rejections often
            # carry the same id).
            for rej in (submission.rejections if submission is not None else []):
                rej_id = str(getattr(rej, "intent_id", "") or "")
                if rej_id and (
                    rej_id.endswith(f":{idx}") or rej_id == intent_id
                ):
                    rejection = rej
                    break

            fill_status = "UNKNOWN"
            filled_shares: float | None = None
            filled_price: float | None = None
            fill_delay_ms: int | None = None
            rejection_reason: str | None = None
            slippage_bps: float | None = None
            post_fill_mid: float | None = None
            audit_extra: dict[str, Any] = {}
            if fill is not None:
                try:
                    filled_shares = float(fill.shares)
                except (TypeError, ValueError):
                    filled_shares = None
                try:
                    filled_price = float(fill.price)
                except (TypeError, ValueError):
                    filled_price = None
                if (
                    filled_shares is not None
                    and submitted_shares is not None
                    and abs(filled_shares - submitted_shares) > 1e-6
                ):
                    fill_status = "PARTIAL"
                else:
                    fill_status = "FILLED"
                if filled_price is not None and submitted_price is not None and submitted_price > 0:
                    slippage_bps = 10_000.0 * (filled_price - submitted_price) / submitted_price
                audit_extra["fill_data"] = {
                    "fill_id": getattr(fill, "fill_id", None),
                    "intent_id": getattr(fill, "intent_id", None),
                    "notional": getattr(fill, "notional", None),
                    "filled_at": (
                        fill.filled_at.isoformat() if getattr(fill, "filled_at", None) is not None
                        else None
                    ),
                }
            elif rejection is not None:
                fill_status = "REJECTED"
                rejection_reason = str(getattr(rejection, "reason", "") or "")
                audit_extra["rejection_intent_id"] = getattr(rejection, "intent_id", None)

            audit_payload: dict[str, Any] = {
                "intent_id": intent_id,
                "tick_ts": tick_id,
                "market_id": intent.market_id,
                "side": intent.side,
                "action": intent.action,
                "submitted_shares": submitted_shares,
                "submitted_price_implied": submitted_price,
                "expected_fill_prob": expected_fill_prob,
                "edge_source": edge_source,
                "fill_status": fill_status,
                "filled_shares": filled_shares,
                "filled_price": filled_price,
                "slippage_bps": slippage_bps,
                "rejection_reason": rejection_reason,
                "quoted_bid_at_decision": quoted_bid,
                "quoted_ask_at_decision": quoted_ask,
                "mid_at_decision": mid,
                "quote_age_sec_at_decision": quote_age,
                **audit_extra,
            }
            rows.append({
                "intent_id": intent_id,
                "tick_ts": tick_id or "",
                "experiment_id": self.experiment_id,
                "participant_idx": self.participant_idx,
                "market_id": intent.market_id,
                "side": intent.side,
                "action": intent.action,
                "quoted_bid_at_decision": quoted_bid,
                "quoted_ask_at_decision": quoted_ask,
                "mid_at_decision": mid,
                "quote_age_sec_at_decision": quote_age,
                "submitted_shares": submitted_shares,
                "submitted_price_implied": submitted_price,
                "edge_source": edge_source,
                "expected_fill_prob": expected_fill_prob,
                "fill_status": fill_status,
                "filled_shares": filled_shares,
                "filled_price": filled_price,
                "fill_delay_ms": fill_delay_ms,
                "rejection_reason": rejection_reason,
                "slippage_bps": slippage_bps,
                "post_fill_mid": post_fill_mid,
                "audit_json": json.dumps(audit_payload, default=str),
                "created_at": created,
            })
        if not rows:
            return
        try:
            self.state.record_fills(rows)
        except Exception as exc:
            self._log(
                "fill_audit_persistence_failed",
                error=_short(exc),
                rows=len(rows),
            )

    def _run_longshot_shadow_pass(self, ctx: Any) -> LongshotPassResult | None:
        """Emit shadow longshot rows for every eligible/blocked candidate."""
        layer = self.layers_config.longshot
        if not layer.enabled:
            self.last_longshot_pass = None
            return None
        conn = None
        if self.state is not None:
            try:
                conn = self.state.connect()
            except Exception as exc:
                self._log("longshot_connect_failed", error=_short(exc))
                conn = None
        try:
            longshot_pass = run_longshot_pass(
                ctx=ctx,
                portfolio=ctx.portfolio,
                candidates=ctx.markets,
                layer=layer,
                conn=conn,
            )
        except Exception as exc:
            self._log("longshot_compute_failed", error=_short(exc))
            self.last_longshot_pass = None
            return None
        self.last_longshot_pass = longshot_pass
        if longshot_pass.shadow_rows and self.state is not None:
            try:
                self.state.record_shadow_proposals(longshot_pass.to_shadow_rows(ctx=ctx))
            except Exception as exc:
                self._log(
                    "longshot_shadow_persistence_failed",
                    error=_short(exc),
                    rows=len(longshot_pass.shadow_rows),
                )
        self._log(
            "longshot_pass",
            summary=longshot_pass.summary,
            primary_enabled=layer.primary_enabled,
        )
        return longshot_pass

    def _log(self, event: str, **fields: Any) -> None:
        if self.experiment_dir is not None:
            with contextlib.suppress(Exception):
                self.experiment_dir.log_event(event, **fields)
        logger.info("%s %s", event, _safe_repr(fields))

    def _shutdown(self) -> None:
        try:
            if self.state is not None:
                with contextlib.suppress(Exception):
                    ckpt = self.state.checkpoint()
                    self._log("shutdown_checkpoint", path=str(ckpt))
                with contextlib.suppress(Exception):
                    self.state.close()
        finally:
            with contextlib.suppress(Exception):
                self.heartbeat.stop()
            if self.session is not None and self._preset_session is None:
                with contextlib.suppress(Exception):
                    self.session.close()
            if self._pid_owned:
                with contextlib.suppress(Exception):
                    release_pid_file(self.pid_path)
                self._pid_owned = False


# --- CLI entrypoint ---------------------------------------------------------


def load_env() -> tuple[str, str]:
    """Resolve PA_SERVER_URL and PA_SERVER_API_KEY without printing secrets."""
    if load_dotenv is not None:
        # Try common locations without surfacing secrets to stdout.
        candidates: list[Path] = [Path(".env"), Path.cwd().parent / ".env"]
        for candidate in candidates:
            with contextlib.suppress(Exception):
                if candidate.is_file():
                    load_dotenv(candidate)
    api_url = os.environ.get("PA_SERVER_URL") or DEFAULT_API_URL
    api_key = os.environ.get("PA_SERVER_API_KEY")
    if not api_key:
        raise SystemExit(
            "PA_SERVER_API_KEY is not set. Source env.sh (or set the variable in .env) "
            "before starting the kalibre loop."
        )
    return api_url, api_key


def configure_logging() -> None:
    level = os.getenv("KALIBRE_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kalibre.loop")
    parser.add_argument("--slug", default=None, help="experiment slug")
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument(
        "--print-restart-command",
        action="store_true",
        help="print the shell command to relaunch with the configured slug and exit",
    )
    args = parser.parse_args(argv)
    configure_logging()
    config = LoopConfig.from_env()
    if args.slug:
        config = LoopConfig(
            slug=args.slug,
            max_ticks=config.max_ticks,
            starting_cash=config.starting_cash,
            submit_cutoff_sec=config.submit_cutoff_sec,
            claim_idle_sleep_sec=config.claim_idle_sleep_sec,
            experiments_dir=config.experiments_dir,
            pid_dir=config.pid_dir,
            heartbeat_interval_sec=config.heartbeat_interval_sec,
            checkpoint_every_n_ticks=config.checkpoint_every_n_ticks,
        )
    if args.max_ticks is not None:
        config = LoopConfig(
            slug=config.slug,
            max_ticks=args.max_ticks,
            starting_cash=config.starting_cash,
            submit_cutoff_sec=config.submit_cutoff_sec,
            claim_idle_sleep_sec=config.claim_idle_sleep_sec,
            experiments_dir=config.experiments_dir,
            pid_dir=config.pid_dir,
            heartbeat_interval_sec=config.heartbeat_interval_sec,
            checkpoint_every_n_ticks=config.checkpoint_every_n_ticks,
        )
    if args.print_restart_command:
        from kalibre.watchdog import build_restart_command
        print(build_restart_command(config.slug))
        return 0

    api_url, api_key = load_env()
    try:
        loop = KalibreLoop(config, api_url=api_url, api_key=api_key)
        return loop.run()
    except AlreadyRunningError as exc:
        sys.stderr.write(f"refusing to start: {exc}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
