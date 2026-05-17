from __future__ import annotations

import os
import subprocess
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
    os.environ.setdefault("PA_MAX_TICKS", "1500")
    os.environ.setdefault("PA_STARTING_CASH", "10000")
    os.environ.setdefault("KALIBRE_STRATEGY_MODE", "forecast_dry_run")
    os.environ.setdefault("KALIBRE_ENABLE_LIVE_TRADES", "1")
    os.environ.setdefault("KALIBRE_MAX_FORECAST_MARKETS", "8")
    os.environ["PA_EXPERIMENT_SLUG"] = _build_slug()


def _start_loop() -> None:
    global RUN_PROC, RUN_META
    with LOCK:
        if RUN_PROC is not None and RUN_PROC.poll() is None:
            return
        _ensure_env_defaults()
        log_path = ROOT / f"kalibre_service_{os.environ['PA_EXPERIMENT_SLUG']}.log"
        log_fp = open(log_path, "a", buffering=1)
        proc = subprocess.Popen(
            ["python", "-m", "kalibre.loop"],
            cwd=str(ROOT),
            env=os.environ.copy(),
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
        )
        RUN_PROC = proc
        RUN_META = {
            "slug": os.environ["PA_EXPERIMENT_SLUG"],
            "log_path": str(log_path),
            "started_at": datetime.now(tz=UTC).isoformat(),
        }


app = FastAPI(title="Kalibre Trading Agent", version="1.0.0")


@app.on_event("startup")
def on_startup() -> None:
    _start_loop()


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

