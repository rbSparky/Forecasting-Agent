"""Halawi-style deterministic prompt builder + JSON response parser.

The prompt forces a four-step reasoning scratchpad (base rate, trend,
recent evidence, confidence-check) but reports only a single JSON line
with the calibrated YES probability so the parser stays stable.

There is **no web search** here. This is Phase 3 forecast core: we only
ask the model to reason from the supplied market context.
"""

from __future__ import annotations

import json
import re
from typing import Any

from kalibre.forecast.types import ForecastRequest


JSON_INSTRUCTION = (
    "Respond with a single JSON object on the last line of your reply, "
    "exactly matching this schema: "
    '{"p_yes": <float in [0, 1]>, "confidence": <float in [0, 1]>, '
    '"rationale": <string under 240 chars>}.'
)

SYSTEM_PROMPT = (
    "You are a calibrated forecaster for binary prediction markets. "
    "You give probabilities that are well-calibrated against historical "
    "base rates and you avoid overconfidence. You never invent facts. "
    "You never browse the web. You work only with the market context "
    "provided to you."
)

SCRATCHPAD_GUIDE = (
    "Work through these four steps internally, then emit the final JSON.\n"
    "1. BASE RATE: state the relevant historical base rate for events of "
    "this kind and any reference class you can think of.\n"
    "2. EXTRAPOLATION: extrapolate that base rate to the specific question.\n"
    "3. RECENT EVIDENCE: weigh anything specific to this market in the "
    "supplied context (rules, resolution time, current quotes).\n"
    "4. CONFIDENCE CHECK: ask whether your probability is unjustifiably "
    "extreme; if so, pull it toward 0.5.\n"
    "You SHOULD NOT include the scratchpad in the output. Only the final "
    "JSON line matters."
)

_CATEGORY_HINTS: dict[str, tuple[str, ...]] = {
    "sports_nfl": ("nfl", "super bowl", "football"),
    "sports_nba": ("nba", "basketball", "finals"),
    "sports_soccer": ("soccer", "champions league", "premier league", "world cup"),
    "sports_nhl": ("nhl", "stanley cup", "hockey"),
    "sports_mlb": ("mlb", "baseball", "world series"),
    "sports_tennis": ("tennis", "wimbledon", "us open", "roland garros"),
    "sports_mma": ("ufc", "mma", "fight"),
    "politics": ("senate", "congress", "nominee", "election", "president"),
    "entertainment": ("oscar", "grammy", "billboard", "streaming", "box office"),
    "companies": ("ipo", "earnings", "revenue", "ceo", "acquisition"),
    "economics": ("cpi", "inflation", "payroll", "fed", "interest rate"),
    "weather_science": ("hurricane", "earthquake", "storm", "temperature", "climate"),
}

_CATEGORY_CHECKLISTS: dict[str, str] = {
    "sports_nfl": (
        "NFL checklist: injuries/inactives and backup quality, line movement vs public splits, "
        "weather (especially wind), rest/travel disadvantage, and matchup trenches."
    ),
    "sports_nba": (
        "NBA checklist: back-to-back / 3-in-4 fatigue, star on/off impact, pace/style matchup, "
        "home-road split, and injury status near tipoff."
    ),
    "sports_soccer": (
        "Soccer checklist: expected-goals form (xG/xGA), projected XI and rotation, fixture congestion, "
        "home-away splits, and tactical matchup."
    ),
    "sports_nhl": (
        "NHL checklist: goalie confirmation/rotation, recent shot quality and expected-goals trends, "
        "schedule fatigue, injuries, and special-teams mismatch."
    ),
    "sports_mlb": (
        "MLB checklist: starting pitcher quality and regression risk, bullpen availability/fatigue, "
        "park/weather effects, lineup handedness splits, and umpire tendencies if relevant."
    ),
    "sports_tennis": (
        "Tennis checklist: surface-specific strength, fatigue from prior match duration, "
        "head-to-head by surface, injury fitness, and court/conditions speed."
    ),
    "sports_mma": (
        "MMA checklist: weigh-in outcome, style matchup (striking/grappling), reach/age edge, "
        "recent form and cardio/training camp signals."
    ),
    "politics": (
        "Politics checklist: procedural path (committee/floor), whip count and coalition math, "
        "public statements, cross-market consensus, and time remaining to resolution."
    ),
    "entertainment": (
        "Entertainment checklist: trajectory math (decay/growth), comparable benchmarks, "
        "calendar catalysts (releases/awards), and platform-specific momentum."
    ),
    "companies": (
        "Companies checklist: event timing certainty, management/board signals, financing/regulatory path, "
        "and comparable precedent outcomes."
    ),
    "economics": (
        "Economics checklist: release schedule and methodology, leading indicators, "
        "consensus vs surprise risk, and revision/seasonality effects."
    ),
    "weather_science": (
        "Weather/science checklist: official methodology and thresholds, base rates, "
        "seasonal regime, and lead-time uncertainty."
    ),
    "default": (
        "General checklist: identify causal drivers, quantify directional impact, "
        "separate knowns from unknowns, and avoid overreacting to narrative noise."
    ),
}


def _infer_category(req: ForecastRequest) -> str:
    hay = " ".join(
        [
            (req.question or ""),
            (req.topic or ""),
            (req.family or ""),
        ],
    ).lower()
    for category, hints in _CATEGORY_HINTS.items():
        if any(h in hay for h in hints):
            return category
    return "default"


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.4f}"


def _fmt_dt(dt: Any) -> str:
    if dt is None:
        return "unknown"
    try:
        return dt.isoformat(timespec="seconds")
    except Exception:
        return str(dt)


def build_prompt(req: ForecastRequest) -> list[dict[str, str]]:
    """Build the OpenAI-style chat messages for one market.

    Returns a list of role/content dicts that an OpenAI-compatible
    ``chat/completions`` endpoint accepts directly.
    """
    spread_str = "n/a" if req.spread is None else f"{req.spread:.4f}"
    volume_str = "n/a" if req.volume_24h is None else f"{req.volume_24h:,.0f}"
    rules_block = f"\nRULES:\n{req.rules}" if req.rules else ""
    description_block = (
        f"\nDESCRIPTION:\n{req.description}" if req.description else ""
    )
    category = _infer_category(req)
    category_checklist = _CATEGORY_CHECKLISTS.get(
        category, _CATEGORY_CHECKLISTS["default"],
    )

    user = (
        f"MARKET: {req.market_id}\n"
        f"QUESTION: {req.question}\n"
        f"SOURCE: {req.source or 'unknown'}\n"
        f"TOPIC: {req.topic or 'unknown'}\n"
        f"FAMILY: {req.family or 'unknown'}\n"
        f"RESOLUTION_TIME_UTC: {_fmt_dt(req.resolution_time)}\n"
        f"BEST_BID: {_fmt_pct(req.best_bid)}\n"
        f"BEST_ASK: {_fmt_pct(req.best_ask)}\n"
        f"MID: {_fmt_pct(req.mid)}\n"
        f"SPREAD: {spread_str}\n"
        f"VOLUME_24H_USD: {volume_str}"
        f"{rules_block}{description_block}\n\n"
        f"CATEGORY_TEMPLATE: {category}\n"
        f"DOMAIN_CHECKLIST: {category_checklist}\n"
        "Use the checklist to structure your internal reasoning. "
        "Do not assume missing facts; if data is missing, reflect uncertainty.\n\n"
        f"{SCRATCHPAD_GUIDE}\n\n{JSON_INSTRUCTION}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


class ForecastParseError(ValueError):
    """Raised when the provider's reply cannot be coerced into a valid forecast."""


def parse_forecast_json(text: str) -> dict[str, Any]:
    """Parse the JSON the model emits. Raises :class:`ForecastParseError`
    on missing / invalid / out-of-range probability."""
    if not text:
        raise ForecastParseError("empty response")
    # The model is asked to emit a JSON object on the last line. Try the
    # last brace-delimited substring first, then fall back to scanning.
    candidates: list[str] = []
    matches = list(_JSON_RE.finditer(text))
    if matches:
        candidates.extend(m.group(0) for m in reversed(matches))
    candidates.append(text.strip())

    last_error: Exception | None = None
    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(data, dict):
            last_error = ForecastParseError("forecast JSON is not an object")
            continue
        if "p_yes" not in data:
            last_error = ForecastParseError("missing 'p_yes'")
            continue
        try:
            p_yes = float(data["p_yes"])
        except (TypeError, ValueError) as exc:
            last_error = ForecastParseError(f"p_yes not numeric: {exc}")
            continue
        if not (0.0 <= p_yes <= 1.0):
            last_error = ForecastParseError(f"p_yes out of [0,1]: {p_yes}")
            continue
        confidence = data.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
                if not (0.0 <= confidence <= 1.0):
                    confidence = None
            except (TypeError, ValueError):
                confidence = None
        rationale = data.get("rationale")
        if rationale is not None and not isinstance(rationale, str):
            rationale = str(rationale)
        return {
            "p_yes": p_yes,
            "confidence": confidence,
            "rationale": (rationale or "")[:480],
        }
    raise ForecastParseError(
        f"could not parse forecast JSON ({type(last_error).__name__ if last_error else 'no candidate'})"
    )
