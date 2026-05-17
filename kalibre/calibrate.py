"""Probability calibration.

Seeds a global calibrator from
``ai-prophet-datasets/datasets/sample-resolved/releases/v1.0.0/tasks.jsonl``.

About the seed dataset: each row carries the resolved outcome for a
multi-outcome task but does *not* include a forecaster's predicted
probability. A true Platt fit needs ``(p_pred, label)`` pairs, which we
don't have at seed time. Phase 3 therefore ships:

- A robust :class:`Calibrator` interface usable at runtime (identity /
  platt / beta).
- An :class:`IdentityCalibrator` factory that records the seed dataset's
  YES base rate so later phases can shrink predictions toward it.
- A Platt fitting routine that future phases can call once we have
  ``(p_pred, outcome)`` pairs from real trading.

Until those pairs exist, ``fit_seed_calibrator`` returns an identity
calibrator with metadata, which is the *documented fallback* mentioned
in the build plan. ``calibrate_probability`` still clips into
``[0.02, 0.98]`` regardless.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("kalibre.calibrate")


P_CAL_LO = 0.02
P_CAL_HI = 0.98


@dataclass
class Calibrator:
    """Serializable calibration model.

    ``method``:

    - ``"identity"`` — return ``clip(p_raw, lo, hi)`` (no learning).
    - ``"platt"``    — ``sigmoid(a * logit(p_raw) + b)`` then clip.
    - ``"beta"``     — Beta calibration (Kull et al. 2017): ``sigmoid(a *
      log(p_raw) + b * log(1 - p_raw) + c)`` then clip.
    """

    method: str = "identity"
    params: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    base_rate_yes: float | None = None
    brier_score: float | None = None
    log_loss: float | None = None
    source: str = "identity_seed"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "Calibrator":
        data = json.loads(raw)
        params = data.get("params") or {}
        return cls(
            method=str(data.get("method", "identity")),
            params={k: float(v) for k, v in params.items()},
            n_samples=int(data.get("n_samples", 0) or 0),
            base_rate_yes=(
                float(data["base_rate_yes"]) if data.get("base_rate_yes") is not None else None
            ),
            brier_score=(float(data["brier_score"]) if data.get("brier_score") is not None else None),
            log_loss=(float(data["log_loss"]) if data.get("log_loss") is not None else None),
            source=str(data.get("source", "identity_seed")),
        )


# --- apply ------------------------------------------------------------------


def _clip(p: float) -> float:
    if p != p:  # NaN
        return 0.5
    return min(P_CAL_HI, max(P_CAL_LO, p))


def _logit(p: float) -> float:
    p = min(0.999_999, max(1e-6, p))
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def calibrate_probability(p_raw: float, calibrator: Calibrator) -> float:
    """Apply ``calibrator`` to ``p_raw`` and clip into ``[0.02, 0.98]``."""
    if p_raw is None:
        return P_CAL_LO
    if calibrator.method == "platt":
        a = calibrator.params.get("a", 1.0)
        b = calibrator.params.get("b", 0.0)
        return _clip(_sigmoid(a * _logit(float(p_raw)) + b))
    if calibrator.method == "beta":
        a = calibrator.params.get("a", 1.0)
        b = calibrator.params.get("b", 1.0)
        c = calibrator.params.get("c", 0.0)
        p = min(0.999_999, max(1e-6, float(p_raw)))
        z = a * math.log(p) + b * math.log(1.0 - p) + c
        return _clip(_sigmoid(z))
    # identity
    return _clip(float(p_raw))


# --- seed ------------------------------------------------------------------


@dataclass
class SeedStats:
    n_tasks: int
    n_yes_events: int
    n_outcomes: int
    base_rate_yes: float
    by_category: dict[str, float]


def load_seed_records(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    records: list[dict[str, Any]] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping malformed JSONL line in %s", p)
    return records


def _yes_count_for_task(task: dict[str, Any]) -> tuple[int, int]:
    """Return ``(n_yes_events, n_outcomes)`` for a sample-resolved task.

    Each outcome in ``outcomes`` is a candidate YES side; the
    ``resolved_outcome.value`` list contains the ones that actually
    happened (one per task in this dataset).
    """
    outcomes = task.get("outcomes") or []
    if not outcomes:
        return 0, 0
    resolved = (task.get("resolved_outcome") or {}).get("value") or []
    resolved_set = {str(v) for v in resolved}
    yes = sum(1 for o in outcomes if str(o) in resolved_set)
    return yes, len(outcomes)


def compute_seed_stats(records: Iterable[dict[str, Any]]) -> SeedStats:
    total_outcomes = 0
    yes_total = 0
    by_cat_yes: dict[str, int] = {}
    by_cat_total: dict[str, int] = {}
    n_tasks = 0
    for task in records:
        yes, n = _yes_count_for_task(task)
        if n == 0:
            continue
        n_tasks += 1
        total_outcomes += n
        yes_total += yes
        category = ((task.get("metadata") or {}).get("category") or "other").lower()
        by_cat_total[category] = by_cat_total.get(category, 0) + n
        by_cat_yes[category] = by_cat_yes.get(category, 0) + yes
    base_rate = yes_total / total_outcomes if total_outcomes else 0.5
    by_category = {
        cat: (by_cat_yes.get(cat, 0) / total) if total else 0.5
        for cat, total in by_cat_total.items()
    }
    return SeedStats(
        n_tasks=n_tasks,
        n_yes_events=yes_total,
        n_outcomes=total_outcomes,
        base_rate_yes=base_rate,
        by_category=by_category,
    )


def fit_seed_calibrator(path: str | Path) -> Calibrator:
    """Build the seed calibrator from sample-resolved.

    Without predicted probabilities, this is an identity calibrator
    seeded with the dataset's YES base rate as metadata. Replacing it
    with a Platt model becomes a one-line swap once Phase 4 / live trading
    produces ``(p_pred, outcome)`` pairs.
    """
    try:
        records = load_seed_records(path)
        stats = compute_seed_stats(records)
    except FileNotFoundError:
        logger.warning("seed dataset not found at %s; using identity calibrator", path)
        return Calibrator(method="identity", source="identity_missing_seed")
    return Calibrator(
        method="identity",
        params={},
        n_samples=stats.n_outcomes,
        base_rate_yes=stats.base_rate_yes,
        source=f"seed:{Path(path).name}",
    )


# --- Platt fitting (for future use) ----------------------------------------


def fit_platt(
    pairs: list[tuple[float, int]],
    *,
    iterations: int = 200,
    lr: float = 0.05,
    l2: float = 1e-4,
) -> Calibrator:
    """Fit a Platt scaler from ``(p_pred, label_0_or_1)`` pairs.

    Plain SGD on the logistic loss. Suitable for the modest sample sizes
    Phase 4+ will collect. If no pairs are supplied, returns identity.
    """
    if not pairs:
        return Calibrator(method="identity", source="platt_no_data")
    a, b = 1.0, 0.0
    for _ in range(iterations):
        ga = 0.0
        gb = 0.0
        for p_pred, label in pairs:
            z = a * _logit(float(p_pred)) + b
            pred = _sigmoid(z)
            err = pred - float(label)
            ga += err * _logit(float(p_pred))
            gb += err
        n = len(pairs)
        a -= lr * (ga / n + l2 * a)
        b -= lr * (gb / n + l2 * b)
    # Compute Brier + log loss for diagnostics.
    brier = 0.0
    log_loss = 0.0
    for p_pred, label in pairs:
        z = a * _logit(float(p_pred)) + b
        pred = _sigmoid(z)
        brier += (pred - label) ** 2
        log_loss += -math.log(max(1e-9, pred if label else 1 - pred))
    n = len(pairs)
    return Calibrator(
        method="platt",
        params={"a": a, "b": b},
        n_samples=n,
        brier_score=brier / n,
        log_loss=log_loss / n,
        source="platt_fit",
    )


# --- load / save -----------------------------------------------------------


def save_calibrator(path: str | Path, calibrator: Calibrator) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(calibrator.to_json())


def load_calibrator(path: str | Path) -> Calibrator | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return Calibrator.from_json(p.read_text())
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("calibrator at %s is unreadable (%s); ignoring", p, exc)
        return None


def load_or_fit_calibrator(
    *,
    cache_path: str | Path,
    seed_path: str | Path,
) -> Calibrator:
    """Return a cached calibrator if available, else fit from the seed."""
    existing = load_calibrator(cache_path)
    if existing is not None:
        return existing
    fitted = fit_seed_calibrator(seed_path)
    try:
        save_calibrator(cache_path, fitted)
    except OSError as exc:
        logger.warning("could not persist seed calibrator: %s", exc)
    return fitted
