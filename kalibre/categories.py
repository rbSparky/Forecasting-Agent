"""Phase 6B deterministic category detection + source-tier classifier.

Pure-function module. No I/O, no LLM. Distilled from
``refined_templates (1).md`` into runtime-only data. The markdown file
is not read at runtime.

Public API:

- :func:`detect_category` -> :class:`CategoryProfile`
- :func:`classify_source_tier` -> ``"A"|"B"|"C"|"D"``

The forecast runner stamps the detected category onto every
``ForecastRequest`` so the prompt's category playbook block and the
audit chain stay coherent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


__all__ = [
    "CategoryProfile",
    "detect_category",
    "classify_source_tier",
    "CATEGORY_KEYWORDS",
    "PLAYBOOK",
    "SOURCE_TIERS",
]


# --- per-category keyword lists -------------------------------------------


CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "nfl": (
        "nfl", "football", "super bowl", "chiefs", "bills", "eagles",
        "cowboys", "49ers", "ravens", "patriots", "packers", "steelers",
        "dolphins", "jets", "giants", "jaguars", "bengals", "broncos",
        "lions", "vikings", "saints", "panthers", "seahawks", "rams",
        "buccaneers", "titans", "colts", "browns", "commanders",
    ),
    "mlb": (
        "mlb", "baseball", "dodgers", "yankees", "astros", "braves",
        "mets", "padres", "red sox", "phillies", "cubs", "cardinals",
        "blue jays", "rays", "guardians", "tigers", "twins", "rangers",
        "diamondbacks", "marlins", "brewers", "rockies", "athletics",
    ),
    "soccer": (
        "soccer", "premier league", "champions league", "arsenal", "psg",
        "man city", "manchester city", "manchester united", "liverpool",
        "real madrid", "barcelona", "bayern", "juventus", "milan", "inter",
        "chelsea", "tottenham", "atletico", "dortmund", "fc",
        "uefa", "la liga", "bundesliga", "serie a", "world cup",
    ),
    "nba": (
        "nba", "basketball", "lakers", "celtics", "warriors", "nuggets",
        "knicks", "thunder", "heat", "bucks", "76ers", "sixers", "suns",
        "mavericks", "clippers", "nets", "raptors", "grizzlies", "pelicans",
        "kings", "magic", "hornets", "bulls", "pistons", "cavaliers",
    ),
    "tennis": (
        "tennis", "wimbledon", "us open", "australian open", "french open",
        "roland garros", "atp", "wta", "alcaraz", "sinner", "djokovic",
        "swiatek", "sabalenka", "gauff", "medvedev",
    ),
    "mma_ufc": (
        "ufc", "mma", "fight", "fighter", "weigh-in", "weigh in", "octagon",
        "knockout", "submission", "bantamweight", "welterweight",
        "heavyweight", "mcgregor", "edwards",
    ),
    "entertainment": (
        "spotify", "billboard", "streaming", "box office", "netflix",
        "album", "song", "movie", "oscar", "grammy", "emmy", "tony",
        "premiere", "blockbuster", "rotten tomatoes",
    ),
    "politics": (
        "senate", "congress", "vote", "election", "president", "governor",
        "nominee", "poll", "committee", "hearing", "primary", "caucus",
        "ballot", "incumbent", "house bill", "impeach", "speaker",
        "appointee", "ratify",
    ),
    "companies": (
        "earnings", "revenue", "ceo", "cfo", "merger", "ipo", "stock",
        "guidance", "shareholder", "tesla", "apple", "google", "amazon",
        "microsoft", "meta", "nvidia", "buyback", "10-k", "10-q",
        "sec filing", "analyst",
    ),
    "mentions": (
        "what will", "say during", "mention", "state the word",
        "use the term", "reference", "say the phrase",
        "utter the word",
    ),
    "economics": (
        "jobs", "payroll", "unemployment", "cpi", "inflation", "gdp",
        "fed", "interest rate", "treasury", "rate cut", "rate hike",
        "ppi", "ism", "nfp", "fomc", "powell",
    ),
    "weather": (
        "temperature", "weather", "forecast", "hurricane", "storm",
        "precipitation", "snowfall", "noaa", "nws", "rainfall",
        "tornado", "blizzard", "high temperature", "low temperature",
    ),
    "nhl": (
        "nhl", "hockey", "goalie", "puck", "stanley cup", "bruins",
        "rangers", "capitals", "oilers", "maple leafs", "lightning",
        "panthers", "blackhawks", "kings", "ducks", "stars", "wild",
    ),
    "crypto": (
        "bitcoin", "ethereum", "crypto", "btc", "eth", "solana", "sol",
        "mvrv", "exchange netflow", "funding rate", "binance", "coinbase",
        "stablecoin", "tether", "usdc", "defi",
    ),
}


# --- per-category playbook ------------------------------------------------


@dataclass(frozen=True)
class CategoryPlaybook:
    """High-signal questions + source quality hints + freshness hints for
    one category. Injected into the prompt as a compact checklist."""

    high_signal_questions: tuple[str, ...]
    source_quality_hints: tuple[str, ...]
    stale_after_hours: float


PLAYBOOK: dict[str, CategoryPlaybook] = {
    "nfl": CategoryPlaybook(
        high_signal_questions=(
            "Are key players listed inactive or questionable?",
            "Has the betting line moved (sharp money) since open?",
            "Weather/wind/precipitation forecast at kickoff?",
            "Rest / travel / short-week situation?",
        ),
        source_quality_hints=(
            "official: nfl.com, team injury reports; B: pff.com, sharrpfootballanalysis.com; "
            "C: espn.com, theathletic.com",
        ),
        stale_after_hours=3.0,
    ),
    "mlb": CategoryPlaybook(
        high_signal_questions=(
            "Probable starting pitchers and recent xFIP/SIERA?",
            "Bullpen usage in last 3 days?",
            "Umpire strike-zone tendencies?",
            "Park factor / weather (wind direction, temp)?",
        ),
        source_quality_hints=(
            "official: mlb.com; B: fangraphs.com, baseball-reference.com, "
            "statcast; C: theathletic.com",
        ),
        stale_after_hours=3.0,
    ),
    "soccer": CategoryPlaybook(
        high_signal_questions=(
            "Predicted lineup + injuries / suspensions?",
            "xG / xGA over last 5 matches?",
            "Rotation due to midweek fixtures?",
            "Motivation / league-table situation?",
        ),
        source_quality_hints=(
            "official: premierleague.com, uefa.com, club sites; "
            "B: fbref.com, understat.com, whoscored.com; C: theguardian.com, bbc.com",
        ),
        stale_after_hours=3.0,
    ),
    "nba": CategoryPlaybook(
        high_signal_questions=(
            "Latest injury report (questionable / out)?",
            "Back-to-back or 3-in-4 fatigue?",
            "On-court / off-court net rating with key absences?",
            "Pace mismatch and recent net rating?",
        ),
        source_quality_hints=(
            "official: nba.com injury report; B: cleaningtheglass.com, bball-index.com; "
            "C: theathletic.com",
        ),
        stale_after_hours=4.0,
    ),
    "tennis": CategoryPlaybook(
        high_signal_questions=(
            "Surface-specific Elo and recent form?",
            "Head-to-head on this surface?",
            "Recent match duration / fatigue?",
            "Injury / withdrawal risk? Court speed / weather?",
        ),
        source_quality_hints=(
            "official: atptour.com, wtatennis.com; B: tennisabstract.com; C: tennis.com",
        ),
        stale_after_hours=6.0,
    ),
    "mma_ufc": CategoryPlaybook(
        high_signal_questions=(
            "Weigh-in result (made weight?)",
            "Reach / age / tale-of-the-tape edge?",
            "Recent performance + training-camp injuries?",
            "Late replacement / fight IQ mismatch?",
        ),
        source_quality_hints=(
            "official: ufc.com; B: tapology.com, mmadecisions.com; "
            "C: mmajunkie.com",
        ),
        stale_after_hours=6.0,
    ),
    "entertainment": CategoryPlaybook(
        high_signal_questions=(
            "Current metric count (streams / sales / box office)?",
            "Comparable release trajectory at the same point?",
            "Decay / growth rate (DoD)?",
            "Competing releases or catalyst / social momentum?",
        ),
        source_quality_hints=(
            "official: billboard.com, boxofficemojo.com; B: variety.com, deadline.com; "
            "C: rollingstone.com",
        ),
        stale_after_hours=12.0,
    ),
    "politics": CategoryPlaybook(
        high_signal_questions=(
            "Committee vote / whip count / quorum?",
            "Public statements from key actors?",
            "Polling averages (RCP / FiveThirtyEight)?",
            "Cross-market price on PredictIt / Polymarket?",
        ),
        source_quality_hints=(
            "official: congress.gov, fec.gov; B: 538.com, realclearpolitics.com; "
            "C: nytimes.com, washingtonpost.com",
        ),
        stale_after_hours=12.0,
    ),
    "companies": CategoryPlaybook(
        high_signal_questions=(
            "Recent earnings transcript / guidance?",
            "Analyst expectations and consensus estimate?",
            "Insider / SEC filings (8-K / Form 4)?",
            "Industry trend / sector flow?",
        ),
        source_quality_hints=(
            "official: sec.gov, company IR; B: bloomberg.com, ft.com, reuters.com; "
            "C: wsj.com, cnbc.com",
        ),
        stale_after_hours=24.0,
    ),
    "mentions": CategoryPlaybook(
        high_signal_questions=(
            "Recent speeches / transcripts on this topic?",
            "Past statements about the term / subject?",
            "Event format / agenda predictability?",
            "Current news cycle pressure to comment?",
        ),
        source_quality_hints=(
            "official: rollcall.com, congress.gov, transcripts; "
            "B: politico.com, axios.com; C: nytimes.com",
        ),
        stale_after_hours=12.0,
    ),
    "economics": CategoryPlaybook(
        high_signal_questions=(
            "Consensus forecast (Reuters / Bloomberg poll)?",
            "Leading indicators (ADP, ISM, GDPNow, CME FedWatch)?",
            "Previous revisions and high-frequency data?",
            "Fed speakers' recent tone?",
        ),
        source_quality_hints=(
            "official: bls.gov, federalreserve.gov, bea.gov; "
            "B: cmegroup.com FedWatch, atlantafed.org/gdpnow; "
            "C: reuters.com, bloomberg.com",
        ),
        stale_after_hours=24.0,
    ),
    "weather": CategoryPlaybook(
        high_signal_questions=(
            "Latest NWS / NOAA forecast for the target zone?",
            "Area Forecast Discussion confidence?",
            "Ensemble / model spread (GFS vs ECMWF)?",
            "Historical normals / records for this date?",
        ),
        source_quality_hints=(
            "official: nws.noaa.gov, weather.gov; B: weather.com, accuweather.com",
        ),
        stale_after_hours=2.0,
    ),
    "nhl": CategoryPlaybook(
        high_signal_questions=(
            "Starting goalie confirmed (morning skate)?",
            "Injuries / scratches?",
            "Rest / back-to-back?",
            "Expected goals / recent form (xG)?",
        ),
        source_quality_hints=(
            "official: nhl.com; B: dailyfaceoff.com, naturalstattrick.com, "
            "moneypuck.com; C: theathletic.com",
        ),
        stale_after_hours=3.0,
    ),
    "crypto": CategoryPlaybook(
        high_signal_questions=(
            "Current price / market cap snapshot?",
            "Funding rates and open-interest skew?",
            "Exchange netflow / reserves?",
            "MVRV / SOPR and macro correlation (DXY, equities)?",
        ),
        source_quality_hints=(
            "B: coinmarketcap.com, glassnode.com, cryptoquant.com, "
            "lookintobitcoin.com; C: coindesk.com, theblock.co",
        ),
        stale_after_hours=2.0,
    ),
    "other": CategoryPlaybook(
        high_signal_questions=(
            "Most authoritative source for this question?",
            "Current news cycle on this exact topic?",
            "Cross-market price (other prediction markets)?",
            "Historical base rate for similar questions?",
        ),
        source_quality_hints=(
            "official: government / official body sites; "
            "B: reuters.com, bloomberg.com, ap.org; C: news outlets",
        ),
        stale_after_hours=24.0,
    ),
}


# --- source tiering -------------------------------------------------------


SourceTier = Literal["A", "B", "C", "D"]


# Tiered domain substrings (lowercase). Longest match wins via sorted order.
SOURCE_TIERS: tuple[tuple[str, SourceTier], ...] = (
    # Tier A: official / data sources.
    ("nws.noaa.gov", "A"), ("weather.gov", "A"), ("noaa.gov", "A"),
    ("bls.gov", "A"), ("federalreserve.gov", "A"), ("bea.gov", "A"),
    ("sec.gov", "A"), ("congress.gov", "A"), ("fec.gov", "A"),
    ("nfl.com", "A"), ("mlb.com", "A"), ("nba.com", "A"), ("nhl.com", "A"),
    ("uefa.com", "A"), ("fifa.com", "A"), ("premierleague.com", "A"),
    ("atptour.com", "A"), ("wtatennis.com", "A"), ("ufc.com", "A"),
    ("billboard.com", "A"), ("boxofficemojo.com", "A"),
    # Tier B: respected aggregators / data sites.
    ("fangraphs.com", "B"), ("baseball-reference.com", "B"),
    ("fbref.com", "B"), ("understat.com", "B"), ("whoscored.com", "B"),
    ("cleaningtheglass.com", "B"), ("bball-index.com", "B"),
    ("tennisabstract.com", "B"), ("tapology.com", "B"),
    ("mmadecisions.com", "B"), ("variety.com", "B"), ("deadline.com", "B"),
    ("realclearpolitics.com", "B"), ("538.com", "B"),
    ("fivethirtyeight.com", "B"), ("cmegroup.com", "B"),
    ("atlantafed.org", "B"), ("dailyfaceoff.com", "B"),
    ("naturalstattrick.com", "B"), ("moneypuck.com", "B"),
    ("coinmarketcap.com", "B"), ("glassnode.com", "B"),
    ("cryptoquant.com", "B"), ("lookintobitcoin.com", "B"),
    ("pff.com", "B"), ("sharpfootballanalysis.com", "B"),
    ("predictit.org", "B"), ("polymarket.com", "B"), ("kalshi.com", "B"),
    # Tier C: news / analysis.
    ("reuters.com", "C"), ("bloomberg.com", "C"), ("ft.com", "C"),
    ("wsj.com", "C"), ("cnbc.com", "C"), ("nytimes.com", "C"),
    ("washingtonpost.com", "C"), ("theathletic.com", "C"),
    ("theguardian.com", "C"), ("bbc.com", "C"), ("ap.org", "C"),
    ("politico.com", "C"), ("axios.com", "C"), ("espn.com", "C"),
    ("coindesk.com", "C"), ("theblock.co", "C"), ("rollingstone.com", "C"),
    ("mmajunkie.com", "C"), ("tennis.com", "C"),
    ("weather.com", "C"), ("accuweather.com", "C"),
    # Tier D: forums / social / unverified.
    ("reddit.com", "D"), ("twitter.com", "D"), ("x.com", "D"),
    ("facebook.com", "D"), ("tiktok.com", "D"), ("4chan.org", "D"),
    ("youtube.com", "D"), ("medium.com", "D"), ("substack.com", "D"),
)


def classify_source_tier(domain: str | None) -> SourceTier:
    """Map a domain to one of A/B/C/D. Unknown -> D (treated as unverified)."""
    if not domain:
        return "D"
    dom = domain.lower().strip()
    # Strip a possible scheme + path so callers can pass either a URL or a host.
    if "//" in dom:
        dom = dom.split("//", 1)[1]
    if "/" in dom:
        dom = dom.split("/", 1)[0]
    if dom.startswith("www."):
        dom = dom[4:]
    for needle, tier in SOURCE_TIERS:
        if needle in dom:
            return tier
    return "D"


# --- detector -------------------------------------------------------------


@dataclass(frozen=True)
class CategoryProfile:
    category: str
    confidence: float
    matched_keywords: tuple[str, ...]
    high_signal_questions: tuple[str, ...] = field(default_factory=tuple)
    source_quality_hints: tuple[str, ...] = field(default_factory=tuple)
    stale_after_hours: float = 24.0


# Stable tie-break priority. Highest priority wins when keyword-hit counts
# match. Mentions is FIRST because it's a structural overlay on top of any
# other category (e.g. "will the senator mention inflation in the speech"
# matches both "mentions" and "economics" -- mentions wins so the prompt
# steers toward speech / transcript evidence, not macro data).
_CATEGORY_PRIORITY: tuple[str, ...] = (
    "mentions", "weather", "economics", "crypto", "companies", "politics",
    "soccer", "nfl", "mlb", "nba", "nhl", "tennis", "mma_ufc",
    "entertainment", "other",
)


_WORD_BOUNDARY = re.compile(r"[a-z0-9]+")


def _normalize(text: str | None) -> str:
    if not text:
        return ""
    return text.lower()


def detect_category(
    question: str | None,
    topic: str | None = None,
    family: str | None = None,
    source: str | None = None,
) -> CategoryProfile:
    """Return the best-matching category, with a list of matched keywords
    and the corresponding playbook hints. Falls back to ``"other"``."""
    haystack = " ".join(
        filter(None, (_normalize(question), _normalize(topic), _normalize(family)))
    )
    if not haystack:
        return _build_profile("other", 0.0, ())

    counts: dict[str, list[str]] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        matches: list[str] = []
        for kw in keywords:
            kw_l = kw.lower()
            if " " in kw_l or "-" in kw_l:
                if kw_l in haystack:
                    matches.append(kw)
            else:
                # Word-boundary match for single tokens to avoid false hits
                # like "nba" inside "anbar".
                if re.search(rf"\b{re.escape(kw_l)}\b", haystack):
                    matches.append(kw)
        if matches:
            counts[category] = matches

    if not counts:
        return _build_profile("other", 0.0, ())

    # Argmax by hit count, ties broken by declared priority.
    max_hits = max(len(v) for v in counts.values())
    tied = [c for c, v in counts.items() if len(v) == max_hits]
    if len(tied) == 1:
        category = tied[0]
    else:
        category = next((c for c in _CATEGORY_PRIORITY if c in tied), tied[0])
    matched = tuple(counts[category])
    confidence = max(0.0, min(1.0, max_hits / 3.0))
    return _build_profile(category, confidence, matched)


def _build_profile(
    category: str, confidence: float, matched: tuple[str, ...],
) -> CategoryProfile:
    playbook = PLAYBOOK.get(category, PLAYBOOK["other"])
    return CategoryProfile(
        category=category,
        confidence=confidence,
        matched_keywords=matched,
        high_signal_questions=playbook.high_signal_questions,
        source_quality_hints=playbook.source_quality_hints,
        stale_after_hours=playbook.stale_after_hours,
    )
