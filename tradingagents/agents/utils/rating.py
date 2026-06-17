"""Shared 5-tier rating vocabulary and a deterministic heuristic parser.

The same five-tier scale (Buy, Overweight, Hold, Underweight, Sell) is used by:
- The Research Manager (investment plan recommendation)
- The Portfolio Manager (final position decision)
- The signal processor (rating extracted for downstream consumers)
- The memory log (rating tag stored alongside each decision entry)

Centralising it here avoids drift between those call sites.
"""

from __future__ import annotations

import re

# Canonical, ordered 5-tier scale (most bullish to most bearish).
RATINGS_5_TIER: tuple[str, ...] = (
    "Buy", "Overweight", "Hold", "Underweight", "Sell",
)

_RATING_SET = {r.lower() for r in RATINGS_5_TIER}

# Matches "Rating: X" / "rating - X" / "Rating: **X**" — tolerates markdown
# bold wrappers and either a colon or hyphen separator.
_RATING_LABEL_RE = re.compile(r"rating.*?[:\-][\s*]*(\w+)", re.IGNORECASE)


def parse_rating(text: str, default: str = "Hold") -> str:
    """Heuristically extract a 5-tier rating from prose text.

    Two-pass strategy:
    1. Look for an explicit "Rating: X" label (tolerant of markdown bold).
    2. Fall back to the first 5-tier rating word found anywhere in the text.

    Returns a Title-cased rating string, or ``default`` if no rating word appears.
    """
    for line in text.splitlines():
        m = _RATING_LABEL_RE.search(line)
        if m and m.group(1).lower() in _RATING_SET:
            return m.group(1).capitalize()

    for line in text.splitlines():
        for word in line.lower().split():
            clean = word.strip("*:.,")
            if clean in _RATING_SET:
                return clean.capitalize()

    return default


# Canonical 3-tier directional scale for the intraday forecaster.
DIRECTIONS_3: tuple[str, ...] = ("Up", "Flat", "Down")

_DIRECTION_SET = {d.lower() for d in DIRECTIONS_3}

# Matches the rendered "Primary signal (next 1h): Up" marker, tolerating the
# markdown bold and the parenthesis/colon punctuation around the direction word.
_DIRECTION_LABEL_RE = re.compile(r"1h\b[^A-Za-z]*\**\s*(Up|Flat|Down)\b", re.IGNORECASE)


def parse_direction(text: str, default: str = "Flat") -> str:
    """Heuristically extract the headline (next-1h) direction from a forecast.

    Two-pass strategy mirroring :func:`parse_rating`:
    1. Look for the rendered "next 1h ... Up/Flat/Down" marker.
    2. Fall back to the first Up/Flat/Down word found anywhere in the text.

    Returns a Title-cased direction string, or ``default`` if none is found.
    """
    for line in text.splitlines():
        m = _DIRECTION_LABEL_RE.search(line)
        if m:
            return m.group(1).capitalize()

    for line in text.splitlines():
        for word in line.lower().split():
            clean = word.strip("*:.,()|")
            if clean in _DIRECTION_SET:
                return clean.capitalize()

    return default
