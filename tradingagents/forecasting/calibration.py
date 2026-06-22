"""Post-hoc confidence calibration for the forecast horizons.

The Portfolio Manager's confidence is a raw LLM guess; on a Flat-dominated,
near-random short-horizon problem it is typically over-confident. This module
fits a monotone map from raw confidence -> empirically-calibrated probability,
per horizon, from the resolved track record (isotonic regression via
pool-adjacent-violators, no scikit-learn dependency), so a chronically
over-confident 70% is displayed as the ~55% it has historically earned.

Critical invariant: forecasts are always LOGGED with RAW confidence. Calibration
is applied only for DISPLAY and for the calibrated-Brier readout, so the map is
always re-fit on raw -> outcome and never trains on its own output. Each horizon
needs at least ``MIN_CALIBRATION_N`` resolved forecasts before it gets a map;
until then ``calibrate`` is the identity, so this is safe to ship before data
accrues.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tradingagents.agents.schemas import FORECAST_HORIZONS, _confidence_band

# A horizon needs at least this many resolved forecasts before its raw confidence
# is remapped; below it, calibration is the identity (avoids overfitting noise).
MIN_CALIBRATION_N = 50

# The fitted map is persisted next to the track-record JSONL with this suffix.
_MAP_SUFFIX = ".calibration.json"


def _pav(values: list[float], weights: list[float] | None = None) -> list[float]:
    """Pool-adjacent-violators: nearest non-decreasing fit to ``values``.

    ``values`` are ordered by ascending raw confidence (already averaged within
    ties; ``weights`` carries the per-value sample counts). Returns one
    non-decreasing fitted value per input.
    """
    if weights is None:
        weights = [1.0] * len(values)
    stack: list[list[float]] = []  # each block: [mean, weight, count]
    for v, w in zip(values, weights, strict=True):
        block = [float(v), float(w), 1]
        while stack and stack[-1][0] >= block[0]:
            pv, pw, pc = stack.pop()
            total = pw + block[1]
            block = [(pv * pw + block[0] * block[1]) / total, total, pc + block[2]]
        stack.append(block)
    fitted: list[float] = []
    for mean, _weight, count in stack:
        fitted.extend([mean] * count)
    return fitted


def _interp(support: list[tuple[float, float]], x: float) -> float:
    """Piecewise-linear lookup of ``x`` against a sorted (raw, calibrated) support."""
    if not support:
        return x
    if x <= support[0][0]:
        return support[0][1]
    if x >= support[-1][0]:
        return support[-1][1]
    for i in range(1, len(support)):
        x0, y0 = support[i - 1]
        x1, y1 = support[i]
        if x <= x1:
            if x1 == x0:
                return y1
            return y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return support[-1][1]


def _fit_support(pairs: list[tuple[float, float]]) -> list[list[float]]:
    """Fit one isotonic map from (raw_prob, outcome) pairs -> [[raw, calibrated], ...].

    Outcomes are averaged within each unique raw confidence (weight = count) before
    the isotonic fit, so all calls made at the same confidence map to one value.
    """
    agg: dict[float, list[float]] = {}
    for x, y in pairs:
        a = agg.setdefault(x, [0.0, 0.0])
        a[0] += y
        a[1] += 1.0
    xs = sorted(agg)
    means = [agg[x][0] / agg[x][1] for x in xs]
    weights = [agg[x][1] for x in xs]
    fitted = _pav(means, weights)
    return [[x, f] for x, f in zip(xs, fitted, strict=True)]


class CalibrationMap:
    """Per-horizon isotonic maps from raw confidence (0-100) to a calibrated one."""

    def __init__(self, supports: dict[str, list[list[float]]] | None = None):
        # horizon -> sorted list of [raw_prob, calibrated_prob] support points
        self.supports = supports or {}

    def is_empty(self) -> bool:
        return not self.supports

    @classmethod
    def fit_from_records(
        cls, records: list[dict], min_n: int = MIN_CALIBRATION_N
    ) -> CalibrationMap:
        """Fit one map per horizon from resolved (confidence, direction_correct) rows."""
        supports: dict[str, list[list[float]]] = {}
        for h, _ in FORECAST_HORIZONS:
            pairs = [
                (r["horizons"][h]["confidence"] / 100.0,
                 1.0 if r["horizons"][h]["direction_correct"] else 0.0)
                for r in records
                if h in r.get("horizons", {})
                and r["horizons"][h].get("direction_correct") is not None
                and r["horizons"][h].get("confidence") is not None
            ]
            if len(pairs) >= min_n:
                supports[h] = _fit_support(pairs)
        return cls(supports)

    def calibrate(self, confidence: int, horizon: str) -> int:
        """Map a raw 0-100 confidence to its calibrated value; identity if no map."""
        support = self.supports.get(horizon)
        if not support:
            return confidence
        cal = _interp([(x, y) for x, y in support], confidence / 100.0)
        return max(0, min(100, round(cal * 100)))

    def to_dict(self) -> dict:
        return self.supports

    @classmethod
    def from_dict(cls, data: dict) -> CalibrationMap:
        return cls({h: [list(p) for p in pts] for h, pts in (data or {}).items()})

    def save(self, path: str | Path) -> None:
        Path(path).expanduser().write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CalibrationMap:
        p = Path(path).expanduser()
        if not p.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return cls()


def _map_path(config: dict | None) -> str | None:
    from tradingagents.forecasting.track_record import _track_path
    log_path = _track_path(config)
    return log_path + _MAP_SUFFIX if log_path else None


def fit_and_save(config: dict | None) -> CalibrationMap:
    """Re-fit the calibration map from the current track record and persist it.

    Always fits on RAW logged confidence, so the map never trains on its own
    output. Returns the (possibly empty) map. Best-effort: never raises.
    """
    from tradingagents.forecasting.track_record import ForecastTrackRecord, _track_path
    try:
        log_path = _track_path(config)
        if not log_path or not Path(log_path).expanduser().exists():
            return CalibrationMap()
        cmap = CalibrationMap.fit_from_records(ForecastTrackRecord(log_path)._load())
        map_path = _map_path(config)
        if map_path and not cmap.is_empty():
            cmap.save(map_path)
        return cmap
    except Exception:
        return CalibrationMap()


def load_map(config: dict | None) -> CalibrationMap:
    """Load the persisted calibration map, or an empty (identity) one. Never raises."""
    path = _map_path(config)
    return CalibrationMap.load(path) if path else CalibrationMap()


_ROW_HORIZON = re.compile(r"^\|\s*Next\s*(\d+[mh])\s*\|", re.IGNORECASE)
_CONF_CELL = re.compile(r"([A-Za-z]+)\s*\((\d+)%\)")


def calibrate_markdown(md: str, cmap: CalibrationMap) -> str:
    """Rewrite the confidence cell of each forecast row with its calibrated value.

    For DISPLAY only — the track record still logs the raw confidence. Leaves the
    markdown untouched when the map is empty. The band word (Low/Medium/High) is
    recomputed from the calibrated value so it stays consistent.
    """
    if not md or cmap.is_empty():
        return md
    out = []
    for line in md.splitlines():
        m = _ROW_HORIZON.match(line)
        if m:
            horizon = m.group(1).lower()

            def repl(mm, _h=horizon):
                cal = cmap.calibrate(int(mm.group(2)), _h)
                return f"{_confidence_band(cal)} ({cal}%)"

            line = _CONF_CELL.sub(repl, line, count=1)
        out.append(line)
    return "\n".join(out)
