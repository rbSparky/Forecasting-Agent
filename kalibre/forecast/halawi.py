"""Halawi-style deterministic prompt builder + JSON response parser.

Phase 6B: the prompt no longer forbids the model from browsing. When the
provider attaches the OpenRouter ``openrouter:web_search`` tool, the
model may issue web searches server-side. The prompt:

- Tells the model it can search.
- Injects a per-category playbook (high-signal questions + tier hints)
  pulled from :mod:`kalibre.categories`.
- Asks for an expanded JSON object with category, evidence_quality,
  base_rate, key_drivers, stale_evidence, market_mid, plus the legacy
  ``p_yes / confidence / rationale``.

The parser is backwards-compatible: a legacy 3-field JSON
``{p_yes, confidence, rationale}`` still validates and the new fields
default to ``None``/empty.
"""

from __future__ import annotations

import json
import re
from typing import Any

from kalibre.forecast.types import ForecastRequest


JSON_INSTRUCTION = (
    "OUTPUT RULES (STRICT):\n"
    "1. The FIRST non-whitespace character of your reply MUST be `{`.\n"
    "2. Emit exactly ONE JSON object. No markdown fences, no prose before "
    "the object, no extra objects after it.\n"
    "3. The minimal valid object is "
    '`{"p_yes":0.42,"confidence":0.62,"rationale":"...","evidence_quality":0.8}`. '
    "All other fields are optional.\n"
    "4. Optional expanded fields (omit if you cannot fill them confidently):\n"
    '   `category` (string), `base_rate` (float|null), '
    '`market_mid` (float), `stale_evidence` (bool), '
    '`key_drivers` (array of {"driver","direction","magnitude_pp"} -- '
    "keep at most 2 items, omit `source_ids` if unsure).\n"
    "5. Keep `rationale` under 300 chars on a single line. No trailing prose."
)

SYSTEM_PROMPT = (
    "You are a careful probabilistic forecaster for prediction markets. "
    "You have access to a server-side web-search tool. Use it sparingly "
    "and only when you need current facts (recent injuries, lineups, "
    "weather, prices, news) that aren't already in the market context. "
    "When you do search, prefer the source tiers listed in the category "
    "playbook. If you cannot find recent evidence, say so via "
    "`stale_evidence: true` and stay close to the market mid / base rate. "
    "You never invent facts. "
    "OUTPUT FORMAT: emit the JSON object FIRST, before any narration. "
    "Keep any prose to <=2 sentences. Identify ONE recent fact (<=48h) "
    "that the market may not yet have fully priced (injury, lineup, line "
    "move >3pp, weather change). If you cannot find one, return market "
    "mid (set `stale_evidence: true`)."
)

SCRATCHPAD_GUIDE = (
    "Work through these four steps internally, then emit the final JSON.\n"
    "1. BASE RATE: state the relevant historical base rate for events of "
    "this kind and any reference class you can think of.\n"
    "2. EXTRAPOLATION: extrapolate that base rate to the specific question.\n"
    "3. RECENT EVIDENCE: weigh anything specific to this market in the "
    "supplied context (rules, resolution time, current quotes) and any "
    "web evidence you retrieved.\n"
    "4. CONFIDENCE CHECK: ask whether your probability is unjustifiably "
    "extreme; if so, pull it toward 0.5.\n"
    "You SHOULD NOT include the scratchpad in the output. Only the final "
    "JSON line matters."
)


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


def _build_category_block(category: str | None) -> str:
    """Return a compact "CATEGORY PLAYBOOK" block for the user prompt.

    Lazily imports kalibre.categories to keep test bootstrapping cheap.
    """
    try:
        from kalibre.categories import PLAYBOOK
    except Exception:
        return ""
    if not category:
        return ""
    book = PLAYBOOK.get(category) or PLAYBOOK.get("other")
    if book is None:
        return ""
    lines = [f"CATEGORY: {category}"]
    if book.high_signal_questions:
        lines.append("HIGH-SIGNAL QUESTIONS (use when searching):")
        for q in book.high_signal_questions:
            lines.append(f"  - {q}")
    if book.source_quality_hints:
        lines.append("SOURCE TIERS:")
        for s in book.source_quality_hints:
            lines.append(f"  - {s}")
    return "\n".join(lines)


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
    category_block = _build_category_block(req.category)
    category_section = f"\n\n{category_block}" if category_block else ""

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
        f"{rules_block}{description_block}"
        f"{category_section}\n\n"
        f"{SCRATCHPAD_GUIDE}\n\n{JSON_INSTRUCTION}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class ForecastParseError(ValueError):
    """Raised when the provider's reply cannot be coerced into a valid forecast."""


def _coerce_float_01(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= f <= 1.0):
        return None
    return f


def _extract_balanced_json_objects(text: str) -> list[str]:
    """Return every brace-balanced JSON-looking substring in ``text``.

    Phase 6B: the expanded schema includes a nested ``key_drivers``
    array of objects, so a simple ``\\{[^{}]*\\}`` regex misses the
    outer object. We walk the string, count braces, and collect any
    balanced top-level ``{...}`` slice. String-literal-aware enough to
    handle escaped quotes inside rationales.
    """
    out: list[str] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        escape = False
        start = i
        while i < n:
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        out.append(text[start:i + 1])
                        i += 1
                        break
            i += 1
        else:
            # Unbalanced; bail.
            break
    return out


def parse_forecast_json(text: str) -> dict[str, Any]:
    """Parse the JSON the model emits. Raises :class:`ForecastParseError`
    on missing / invalid / out-of-range probability.

    Returned keys (Phase 6B):

    - ``p_yes`` (required)
    - ``confidence`` (optional, may be ``None``)
    - ``rationale`` (optional, may be ``""``)
    - ``evidence_quality`` (optional, may be ``None``)
    - ``category`` (optional, may be ``None``)
    - ``base_rate`` (optional, may be ``None``)
    - ``market_mid`` (optional, may be ``None``)
    - ``key_drivers`` (optional, may be ``[]``)
    - ``stale_evidence`` (optional, may be ``False``)
    """
    if not text:
        raise ForecastParseError("empty response")
    # The model is asked to emit a JSON object on the last line. Walk
    # the reply for brace-balanced JSON candidates, then try the LAST
    # one first (refined drafts override earlier ones), longest tied.
    candidates: list[str] = []
    balanced = _extract_balanced_json_objects(text)
    if balanced:
        # Reverse order so the last balanced object is tried first.
        candidates.extend(reversed(balanced))
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
        confidence = _coerce_float_01(data.get("confidence")) if data.get("confidence") is not None else None
        rationale = data.get("rationale")
        if rationale is not None and not isinstance(rationale, str):
            rationale = str(rationale)
        # Phase 6B fields (all optional, all backwards-compatible).
        evidence_quality = (
            _coerce_float_01(data.get("evidence_quality"))
            if data.get("evidence_quality") is not None else None
        )
        category = data.get("category")
        if category is not None and not isinstance(category, str):
            category = str(category)
        base_rate = (
            _coerce_float_01(data.get("base_rate"))
            if data.get("base_rate") is not None else None
        )
        market_mid = (
            _coerce_float_01(data.get("market_mid"))
            if data.get("market_mid") is not None else None
        )
        key_drivers = data.get("key_drivers")
        if not isinstance(key_drivers, list):
            key_drivers = []
        stale_evidence = data.get("stale_evidence")
        if not isinstance(stale_evidence, bool):
            stale_evidence = False
        return {
            "p_yes": p_yes,
            "confidence": confidence,
            "rationale": (rationale or "")[:480],
            "evidence_quality": evidence_quality,
            "category": category,
            "base_rate": base_rate,
            "market_mid": market_mid,
            "key_drivers": key_drivers,
            "stale_evidence": stale_evidence,
        }
    raise ForecastParseError(
        f"could not parse forecast JSON ({type(last_error).__name__ if last_error else 'no candidate'})"
    )
