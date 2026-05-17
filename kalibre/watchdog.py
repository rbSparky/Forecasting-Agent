"""Single-writer enforcement, heartbeat, watchdog.

T0 keeps everything strictly in-process. The components:

- :func:`ensure_single_writer`: refuse to start if another live PID owns
  the slug.
- :class:`Heartbeat`: writes a timestamp file every ``interval_sec`` so
  an external watchdog (or :func:`watchdog_check`) can detect a stuck
  process.
- :func:`watchdog_check`: returns ``True`` if the heartbeat is fresh.
- :func:`build_restart_command`: print the exact shell line to relaunch
  the loop for a slug.

systemd integration is left to ops: the unit just needs to run the
restart command with ``Restart=always`` and pass the slug via the
environment.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import sys
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


class AlreadyRunningError(RuntimeError):
    """Another live process already owns this PID file."""


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by another user; treat as alive.
        return True
    except OSError:
        return False
    return True


def ensure_single_writer(pid_path: Path) -> int:
    """Claim ``pid_path`` for the current process. Raises if taken."""
    pid_path = Path(pid_path)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    if pid_path.exists():
        existing: int | None = None
        with contextlib.suppress(Exception):
            existing = int(pid_path.read_text().strip() or 0)
        if existing and existing != os.getpid() and _pid_alive(existing):
            raise AlreadyRunningError(
                f"PID file {pid_path} is owned by live process {existing}"
            )
    pid = os.getpid()
    pid_path.write_text(f"{pid}\n")
    return pid


def release_pid_file(pid_path: Path) -> None:
    pid_path = Path(pid_path)
    if not pid_path.exists():
        return
    with contextlib.suppress(Exception):
        existing = int(pid_path.read_text().strip() or 0)
        if existing == os.getpid():
            pid_path.unlink()


@dataclass
class Heartbeat:
    """Periodic heartbeat writer.

    The path is rewritten with the current UTC timestamp every
    ``interval_sec``. Call :meth:`start` once before the main loop and
    :meth:`stop` on shutdown.
    """

    path: Path
    interval_sec: float = 60.0
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def beat_once(self, *, when: datetime | None = None) -> None:
        ts = (when or datetime.now(tz=UTC)).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp.write_text(ts + "\n")
            tmp.replace(self.path)
        except OSError:
            with contextlib.suppress(OSError):
                self.path.write_text(ts + "\n")

    def start(self) -> None:
        self._stop_event.clear()
        self.beat_once()
        thread = threading.Thread(target=self._loop, name="kalibre-heartbeat", daemon=True)
        thread.start()
        self._thread = thread

    def stop(self) -> None:
        self._stop_event.set()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            with contextlib.suppress(Exception):
                self.beat_once()
            self._stop_event.wait(self.interval_sec)


def watchdog_check(heartbeat_path: Path, max_age_sec: float = 180.0) -> bool:
    """Return ``True`` if the heartbeat file is younger than ``max_age_sec``."""
    heartbeat_path = Path(heartbeat_path)
    if not heartbeat_path.exists():
        return False
    try:
        raw = heartbeat_path.read_text().strip()
    except OSError:
        return False
    if not raw:
        return False
    try:
        ts = datetime.fromisoformat(raw.splitlines()[0])
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    age = (datetime.now(tz=UTC) - ts).total_seconds()
    return 0 <= age <= max_age_sec


def build_restart_command(slug: str, *, python: str | None = None) -> str:
    """Return the exact shell command to relaunch the loop for ``slug``."""
    py = python or sys.executable
    return (
        f"PA_EXPERIMENT_SLUG={shlex.quote(slug)} "
        f"{shlex.quote(py)} -m kalibre.loop"
    )


# --- non-destructive monitor ------------------------------------------------


@dataclass(frozen=True)
class WatchdogStatus:
    """Snapshot of process health, suitable for a non-destructive monitor.

    Fields:

    - ``healthy``: True when the PID is alive *and* the heartbeat is
      fresh.
    - ``pid``: PID read from the pid file, if any.
    - ``pid_alive``: whether that PID currently exists.
    - ``heartbeat_age_sec``: seconds since the last heartbeat, or None.
    - ``stale``: True when the heartbeat is missing or older than
      ``max_age_sec``.
    - ``action``: one of ``"ok"``, ``"restart_required"``, or
      ``"missing"``. ``"missing"`` means no PID file and no heartbeat;
      the loop was probably never started for this slug.
    - ``restart_command``: shell command the operator can run to recover,
      or None when no action is required.
    """

    healthy: bool
    pid: int | None
    pid_alive: bool
    heartbeat_age_sec: float | None
    stale: bool
    action: str
    restart_command: str | None


def _read_pid_file(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    try:
        raw = pid_path.read_text().strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        pid = int(raw.splitlines()[0])
    except ValueError:
        return None
    return pid if pid > 0 else None


def _heartbeat_age_sec(
    heartbeat_path: Path, *, now: datetime | None = None,
) -> float | None:
    if not heartbeat_path.exists():
        return None
    try:
        raw = heartbeat_path.read_text().strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw.splitlines()[0])
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    now = now or datetime.now(tz=UTC)
    return (now - ts).total_seconds()


def watchdog_monitor(
    pid_path: Path,
    heartbeat_path: Path,
    slug: str,
    *,
    max_age_sec: float = 180.0,
    now: datetime | None = None,
) -> WatchdogStatus:
    """Inspect PID + heartbeat state for ``slug`` without killing anything.

    Returns a :class:`WatchdogStatus`. The caller decides whether to act
    on ``restart_required`` (e.g., notify, SIGTERM, ``systemctl restart``).

    Non-destructive by design: this function only reads files and signals
    a 0-check against the PID. It never sends a real kill signal.
    """
    pid = _read_pid_file(Path(pid_path))
    pid_alive = pid is not None and _pid_alive(pid)
    age = _heartbeat_age_sec(Path(heartbeat_path), now=now)
    stale = age is None or age > max_age_sec
    restart = build_restart_command(slug)

    if pid is None and age is None:
        # Nothing on disk - the loop was never started for this slug.
        return WatchdogStatus(
            healthy=False,
            pid=None,
            pid_alive=False,
            heartbeat_age_sec=None,
            stale=True,
            action="missing",
            restart_command=restart,
        )

    if pid_alive and not stale:
        return WatchdogStatus(
            healthy=True,
            pid=pid,
            pid_alive=True,
            heartbeat_age_sec=age,
            stale=False,
            action="ok",
            restart_command=None,
        )

    # Any combination of stale heartbeat or dead PID warrants a restart
    # recommendation. The caller decides whether to take it.
    return WatchdogStatus(
        healthy=False,
        pid=pid,
        pid_alive=pid_alive,
        heartbeat_age_sec=age,
        stale=stale,
        action="restart_required",
        restart_command=restart,
    )
