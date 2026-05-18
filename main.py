from __future__ import annotations

import os
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI


ROOT = Path(__file__).resolve().parent
RUN_PROC: subprocess.Popen[str] | None = None
RUN_META: dict[str, Any] = {}
LOCK = threading.Lock()


def _slugify_team_name(raw: str | None) -> str:
    text = (raw or "team").strip().lower()
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in text)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or "team"


def _build_slug() -> str:
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    team = _slugify_team_name(os.getenv("KALIBRE_TEAM_NAME"))
    user_slug = (os.getenv("PA_EXPERIMENT_SLUG") or "").strip()
    if user_slug:
        return user_slug if user_slug.startswith("eval_") else f"eval_{team}-{user_slug}"
    return f"eval_{team}-{ts}"


def _ensure_env_defaults() -> None:
    # Eval-run defaults.
    os.environ.setdefault("PA_MAX_TICKS", "1500")
    os.environ.setdefault("PA_STARTING_CASH", "10000")
    os.environ.setdefault("KALIBRE_STRATEGY_MODE", "forecast_dry_run")
    os.environ.setdefault("KALIBRE_ENABLE_LIVE_TRADES", "1")
    os.environ.setdefault("KALIBRE_MAX_FORECAST_MARKETS", "8")

    # Phase 6D/6E validated-pipeline defaults. ``setdefault`` preserves
    # any value the Render dashboard / operator sets at deploy time.
    # Budget envelope: $50 total / $3.25 daily / $0.05 per-tick paid cap.
    os.environ.setdefault("KALIBRE_BUDGET_PROFILE", "micro")
    # Forecast routing.
    os.environ.setdefault("KALIBRE_WEB_SEARCH_MODE", "1")
    os.environ.setdefault("KALIBRE_SELECTOR_EXPLORATION_MODE", "1")
    # Opus escalation: enabled but profile-gated. Under the micro
    # profile it only fires when blended edge >=
    # KALIBRE_AUTO_OPUS_MIN_EDGE_PP (default 1.0pp).
    os.environ.setdefault("KALIBRE_OPUS_ESCALATION_MODE", "1")
    os.environ.setdefault("KALIBRE_OPUS_MAX_CALLS_PER_TICK", "1")
    # Live canary safety caps. Without these, intents bypass the
    # per-trade clip and submit at full Kelly size.
    os.environ.setdefault("KALIBRE_LIVE_CANARY_MODE", "1")
    os.environ.setdefault("KALIBRE_CANARY_MAX_INTENTS", "1")
    os.environ.setdefault("KALIBRE_CANARY_MAX_SIZE_USD", "25")
    # Kelly-rescue promotion: the path that produced actionable
    # candidates in pre-deploy testing. Without this, rescue-eligible
    # markets emit shadow rows only.
    os.environ.setdefault("KALIBRE_CANARY_KELLY_RESCUE_MODE", "1")
    os.environ.setdefault("KALIBRE_CANARY_RESCUE_MIN_EDGE_PP", "1.0")

    os.environ["PA_EXPERIMENT_SLUG"] = _build_slug()


def _start_loop() -> None:
    global RUN_PROC, RUN_META
    with LOCK:
        if RUN_PROC is not None and RUN_PROC.poll() is None:
            return
        _ensure_env_defaults()
        # Pipe the loop's stdout/stderr through the parent process so
        # Render's Logs tab streams the actual kalibre output. Without
        # this, all forecast / spend / error logs hide in a file inside
        # the ephemeral container filesystem.
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        proc = subprocess.Popen(
            ["python", "-u", "-m", "kalibre.loop"],
            cwd=str(ROOT),
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
        )
        RUN_PROC = proc
        RUN_META = {
            "slug": os.environ["PA_EXPERIMENT_SLUG"],
            "started_at": datetime.now(tz=UTC).isoformat(),
        }


def _supervisor_loop() -> None:
    """Resurrect the kalibre subprocess if it dies.

    Render Starter (512MB) can OOM-kill the loop mid web-search.
    Without a supervisor, FastAPI keeps serving /healthz while the bot
    is dead. Poll every 30s; respawn if poll() returns non-None.
    """
    while True:
        try:
            with LOCK:
                dead = RUN_PROC is None or RUN_PROC.poll() is not None
            if dead:
                print(
                    f"[supervisor] subprocess dead "
                    f"(exit_code={RUN_PROC.poll() if RUN_PROC else None}), respawning",
                    flush=True,
                )
                _start_loop()
        except Exception as exc:
            print(f"[supervisor] error: {exc}", flush=True)
        threading.Event().wait(30)


app = FastAPI(title="Kalibre Trading Agent", version="1.0.0")


@app.on_event("startup")
def on_startup() -> None:
    _start_loop()
    # Background supervisor: respawn the kalibre subprocess if Render
    # OOM-kills it or it crashes. Without this, a single failure leaves
    # the bot dead for the rest of the eval window.
    threading.Thread(
        target=_supervisor_loop, name="kalibre-supervisor", daemon=True,
    ).start()


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    with LOCK:
        running = RUN_PROC is not None and RUN_PROC.poll() is None
        pid = RUN_PROC.pid if RUN_PROC is not None else None
        exit_code = None if running else (RUN_PROC.poll() if RUN_PROC is not None else None)
        return {
            "status": "ok" if running else "degraded",
            "running": running,
            "pid": pid,
            "exit_code": exit_code,
            **RUN_META,
        }


@app.post("/restart")
def restart() -> dict[str, Any]:
    global RUN_PROC
    with LOCK:
        if RUN_PROC is not None and RUN_PROC.poll() is None:
            RUN_PROC.terminate()
            RUN_PROC.wait(timeout=30)
        RUN_PROC = None
    _start_loop()
    return {"status": "restarted", **RUN_META}

