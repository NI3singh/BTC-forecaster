"""Forecast track record: log every 1h/4h forecast and score it once the horizon elapses.

This is the honesty layer (and the eventual sales proof): an append-only JSONL
log of every forecast, resolved against the realized hourly close, with a rolling
accuracy summary (directional hit-rate, range coverage, mean absolute % error).

The core (``parse_forecast_markdown`` + ``ForecastTrackRecord``) is pure logic —
the realized-price lookup is injected as a callback, so it is fully testable
without network access. The ``record_forecast`` / ``score_and_summarize`` helpers
at the bottom are the thin wiring that backs the callback with the live data
layer; both are best-effort and never raise into the caller.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Horizon label -> hours ahead. Mirrors config["forecast_horizons"] = [1, 4].
HORIZON_HOURS: dict[str, int] = {"1h": 1, "4h": 4}

# A realized move smaller than this fraction of the entry price counts as "Flat".
DEADBAND = 0.001  # 0.1%, ~ round-trip fees

# One rendered forecast table row, e.g.
#   | Next 1h | Up | $65,950.00 | $65,700 - $66,150 | Medium (62%) |
_ROW_RE = re.compile(
    r"\|\s*Next\s*(?P<h>\d+h)\s*\|\s*(?P<dir>Up|Flat|Down)\s*\|\s*"
    r"\$?(?P<price>[\d,]+\.?\d*)\s*\|\s*(?P<range>[^|]+?)\s*\|\s*"
    r"[A-Za-z]+\s*\((?P<conf>\d+)%\)\s*\|",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(r"\$?(?P<lo>[\d,]+\.?\d*)\s*-\s*\$?(?P<hi>[\d,]+\.?\d*)")


def _to_float(s: str) -> float:
    return float(s.replace(",", "").replace("$", "").strip())


def parse_forecast_markdown(md: str) -> dict:
    """Extract per-horizon predictions from a ``render_forecast()`` markdown table.

    Returns ``{"1h": {direction, expected_price, range_low, range_high,
    confidence}, "4h": {...}}``. Rows that don't parse are skipped; an empty
    dict means nothing was found.
    """
    out: dict[str, dict] = {}
    for m in _ROW_RE.finditer(md or ""):
        rng = m.group("range").strip()
        rlo = rhi = None
        rm = _RANGE_RE.search(rng)
        if rm:
            rlo, rhi = _to_float(rm.group("lo")), _to_float(rm.group("hi"))
        out[m.group("h").lower()] = {
            "direction": m.group("dir").capitalize(),
            "expected_price": _to_float(m.group("price")),
            "range_low": rlo,
            "range_high": rhi,
            "confidence": int(m.group("conf")),
        }
    return out


def realized_direction(entry: float, realized: float, deadband: float = DEADBAND) -> str:
    """Classify the realized move from ``entry`` to ``realized`` as Up/Flat/Down."""
    if realized > entry * (1 + deadband):
        return "Up"
    if realized < entry * (1 - deadband):
        return "Down"
    return "Flat"


class ForecastTrackRecord:
    """Append-only JSONL store of forecasts plus their realized outcomes."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def _write(self, records: list[dict]) -> None:
        # Atomic temp-file swap so a crash mid-write never corrupts the log.
        tmp = self.path.with_suffix(".tmp")
        body = "\n".join(json.dumps(r) for r in records)
        tmp.write_text(body + ("\n" if records else ""), encoding="utf-8")
        tmp.replace(self.path)

    def log(self, asset: str, as_of_iso: str, predictions: dict,
            current_price: float | None = None, model_version: str = "") -> bool:
        """Append a forecast. Idempotent on ``(asset, as_of_iso)``; returns True if written."""
        records = self._load()
        for r in records:
            if r["asset"] == asset and r["as_of"] == as_of_iso:
                return False
        records.append({
            "asset": asset,
            "as_of": as_of_iso,
            "current_price": current_price,
            "model_version": model_version,
            "horizons": {
                h: {**p, "realized_price": None, "direction_correct": None,
                    "in_range": None, "abs_pct_error": None}
                for h, p in (predictions or {}).items()
            },
        })
        self._write(records)
        return True

    def resolve(self, price_at: Callable[[str, str], float | None]) -> int:
        """Score elapsed horizons. ``price_at(asset, target_iso)`` returns the realized
        close, or ``None`` when that bar hasn't elapsed / isn't available yet.
        Returns the number of horizons newly scored.
        """
        records = self._load()
        scored = 0
        for r in records:
            entry = r.get("current_price")
            for h, p in r.get("horizons", {}).items():
                if p.get("realized_price") is not None:
                    continue
                hours = HORIZON_HOURS.get(h)
                if hours is None:
                    continue
                target = (datetime.fromisoformat(r["as_of"]) + timedelta(hours=hours)).isoformat()
                realized = price_at(r["asset"], target)
                if realized is None:
                    continue
                p["realized_price"] = realized
                if entry:
                    p["direction_correct"] = realized_direction(entry, realized) == p["direction"]
                    p["abs_pct_error"] = abs(realized - p["expected_price"]) / entry
                if p.get("range_low") is not None and p.get("range_high") is not None:
                    p["in_range"] = p["range_low"] <= realized <= p["range_high"]
                scored += 1
        if scored:
            self._write(records)
        return scored

    def summary(self) -> dict:
        """Aggregate resolved horizons into per-horizon accuracy stats."""
        records = self._load()
        by_horizon: dict[str, dict] = {}
        for h in HORIZON_HOURS:
            rows = [r["horizons"][h] for r in records
                    if h in r.get("horizons", {})
                    and r["horizons"][h].get("realized_price") is not None]
            dir_rows = [x for x in rows if x.get("direction_correct") is not None]
            rng_rows = [x for x in rows if x.get("in_range") is not None]
            err_rows = [x for x in rows if x.get("abs_pct_error") is not None]
            by_horizon[h] = {
                "resolved": len(rows),
                "directional_accuracy": (
                    sum(1 for x in dir_rows if x["direction_correct"]) / len(dir_rows)
                    if dir_rows else None
                ),
                "range_coverage": (
                    sum(1 for x in rng_rows if x["in_range"]) / len(rng_rows)
                    if rng_rows else None
                ),
                "mean_abs_pct_error": (
                    sum(x["abs_pct_error"] for x in err_rows) / len(err_rows)
                    if err_rows else None
                ),
            }
        return {"total_forecasts": len(records), "by_horizon": by_horizon}

    def summary_markdown(self) -> str:
        s = self.summary()

        def pct(v):
            return "n/a" if v is None else f"{v * 100:.0f}%"

        lines = [
            f"## Forecast track record ({s['total_forecasts']} forecasts logged)",
            "",
            "| Horizon | Resolved | Directional acc. | Range coverage | Mean abs % err |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for h, st in s["by_horizon"].items():
            lines.append(
                f"| {h} | {st['resolved']} | {pct(st['directional_accuracy'])} | "
                f"{pct(st['range_coverage'])} | {pct(st['mean_abs_pct_error'])} |"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Wiring helpers (back the price_at callback with the live data layer).
# Best-effort: these never raise into the caller, so a track-record hiccup can
# never break a forecast run.
# ---------------------------------------------------------------------------


def _track_path(config: dict | None) -> str | None:
    from tradingagents.dataflows.config import get_config
    cfg = config or {}
    return cfg.get("forecast_log_path") or get_config().get("forecast_log_path")


def record_forecast(config: dict | None, asset: str, as_of_date: str,
                    forecast_markdown: str, model_version: str = "agents-v1-no-kronos") -> bool:
    """Parse a rendered forecast and log it with the as_of spot price. Never raises."""
    try:
        import pandas as pd

        from tradingagents.dataflows.stockstats_utils import load_ohlcv
        preds = parse_forecast_markdown(forecast_markdown)
        path = _track_path(config)
        if not preds or not path:
            return False
        df = load_ohlcv(asset, as_of_date)
        if df is None or df.empty:
            return False
        last = df.iloc[-1]
        as_of_iso = pd.to_datetime(last["Date"]).isoformat()
        current_price = float(last["Close"])
        return ForecastTrackRecord(path).log(
            asset, as_of_iso, preds, current_price, model_version
        )
    except Exception:
        return False


def build_price_at() -> Callable[[str, str], float | None]:
    """Realized-price lookup backed by the hourly data layer (per-asset cached)."""
    import pandas as pd

    from tradingagents.dataflows.stockstats_utils import load_ohlcv
    cache: dict[str, object] = {}

    def price_at(asset: str, target_iso: str) -> float | None:
        try:
            df = cache.get(asset)
            if df is None:
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                d = load_ohlcv(asset, today)
                d = d.assign(_ts=pd.to_datetime(d["Date"])).set_index("_ts").sort_index()
                cache[asset] = d
                df = d
            target = pd.to_datetime(target_iso).floor("h")
            prior = df[df.index <= target]
            later = df[df.index > target]
            # Only score once the horizon has actually elapsed (a later bar exists).
            if prior.empty or later.empty:
                return None
            return float(prior["Close"].iloc[-1])
        except Exception:
            return None

    return price_at


def score_and_summarize(config: dict | None) -> str:
    """Resolve elapsed forecasts and return the rolling accuracy summary (markdown)."""
    path = _track_path(config)
    if not path:
        return "No forecast_log_path configured."
    tr = ForecastTrackRecord(path)
    tr.resolve(build_price_at())
    return tr.summary_markdown()
