"""Forecast track record: log every intraday multi-horizon forecast and score it as each horizon elapses.

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
import math
import re
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tradingagents.agents.schemas import FORECAST_HORIZONS

# Horizon label -> minutes ahead, derived from the canonical FORECAST_HORIZONS
# (schemas.py) so the schema, renderer, parser, and scorer never drift apart.
HORIZON_MINUTES: dict[str, int] = dict(FORECAST_HORIZONS)

# A realized move smaller than this fraction of the entry price counts as "Flat".
# This is the band at the REFERENCE horizon; ``deadband_for`` scales it with the
# square root of time so longer horizons get a proportionally wider Flat band.
DEADBAND = 0.001  # 0.1%, ~ round-trip fees

# Horizon (minutes) at which DEADBAND applies as-is. A fixed band means different
# things at different horizons — 0.1% swallows most genuine 5m moves as "Flat"
# while being trivially exceeded at 4h — so the band is scaled from this anchor.
DEADBAND_REF_MINUTES = 5

# Below this many resolved rows, a per-horizon accuracy is flagged small-sample in
# the summary so early noise is not mistaken for measured signal.
MIN_RESOLVED_N = 30

# A horizon needs at least this many resolved forecasts before its outcomes are fed
# back into the Portfolio Manager prompt, so a young log can't anchor it to noise.
_FEEDBACK_MIN_RESOLVED = 10

# Trailing 5m bars used to estimate realized per-bar volatility (sigma) and recent
# swing levels for the Portfolio Manager's price anchor and vol-scaled ranges.
_SIGMA_WINDOW_BARS = 100
_SIGMA_MIN_BARS = 20

# One rendered forecast table row, e.g.
#   | Next 1h | Up | $65,950.00 | $65,700 - $66,150 | Medium (62%) |
_ROW_RE = re.compile(
    r"\|\s*Next\s*(?P<h>\d+[mh])\s*\|\s*(?P<dir>Up|Flat|Down)\s*\|\s*"
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


def deadband_for(
    minutes: int, base: float = DEADBAND, ref_minutes: int = DEADBAND_REF_MINUTES
) -> float:
    """Horizon-scaled Flat band: ``base * sqrt(minutes / ref_minutes)``.

    A single fixed band is incoherent across horizons — 0.1% labels most genuine
    5m moves "Flat" yet is trivially exceeded at 4h, so the Up/Flat/Down classes
    mean different things at each horizon and per-horizon accuracy is not
    comparable. Scaling by the square root of time (how a random walk's spread
    grows) keeps "Flat = inside the noise" consistent from 5m to 4h, anchored so
    the shortest horizon keeps the fee-floor ``base``.
    """
    if minutes <= 0:
        return base
    return base * math.sqrt(minutes / ref_minutes)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (pure, no scipy).

    Small live samples make a bare hit-rate look precise when it is not: 6/10 is
    60% but the truth plausibly spans ~30%-83%. The interval is the honest band.
    Trials are assumed iid Bernoulli; the six horizons of one forecast are
    serially correlated, so the true interval is somewhat wider than this.
    """
    if n <= 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


# Confidence bands for the reliability curve (low inclusive, high exclusive; the
# last bin is closed at 100). Coarse on purpose so per-bin counts stay meaningful.
_RELIABILITY_EDGES = [(0, 50), (50, 60), (60, 70), (70, 80), (80, 101)]


def _reliability_bins(pairs: list[tuple[float, float]]) -> list[dict]:
    """Bucket (predicted_prob, outcome) pairs into a reliability curve: per band,
    the mean predicted probability vs the empirical hit-rate, plus the count.

    A well-calibrated forecaster has mean_predicted ~= empirical_accuracy in every
    band; a persistent gap (e.g. 70-80% calls only right 52%) is overconfidence.
    """
    bins = []
    for lo, hi in _RELIABILITY_EDGES:
        sel = [(p, o) for p, o in pairs if lo / 100.0 <= p < hi / 100.0]
        if not sel:
            continue
        bins.append({
            "band": f"{lo}-{min(hi, 100)}%",
            "n": len(sel),
            "mean_predicted": sum(p for p, _ in sel) / len(sel),
            "empirical_accuracy": sum(o for _, o in sel) / len(sel),
        })
    return bins


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

    def resolve(
        self,
        price_at: Callable[[str, str], float | None],
        deadband_base: float = DEADBAND,
    ) -> int:
        """Score elapsed horizons. ``price_at(asset, target_iso)`` returns the realized
        close, or ``None`` when that bar hasn't elapsed / isn't available yet.
        Returns the number of horizons newly scored.

        Each horizon is graded with a volatility-scaled Flat band (``deadband_for``)
        and the band used is stored on the row, so a row stays self-consistent (and
        its baselines stay comparable) even if ``deadband_base`` is later retuned.
        """
        records = self._load()
        scored = 0
        for r in records:
            entry = r.get("current_price")
            for h, p in r.get("horizons", {}).items():
                if p.get("realized_price") is not None:
                    continue
                minutes = HORIZON_MINUTES.get(h)
                if minutes is None:
                    continue
                target = (datetime.fromisoformat(r["as_of"]) + timedelta(minutes=minutes)).isoformat()
                realized = price_at(r["asset"], target)
                if realized is None:
                    continue
                db = deadband_for(minutes, base=deadband_base)
                p["realized_price"] = realized
                p["deadband"] = db
                if entry:
                    p["direction_correct"] = realized_direction(entry, realized, db) == p["direction"]
                    p["abs_pct_error"] = abs(realized - p["expected_price"]) / entry
                if p.get("range_low") is not None and p.get("range_high") is not None:
                    p["in_range"] = p["range_low"] <= realized <= p["range_high"]
                scored += 1
        if scored:
            self._write(records)
        return scored

    @staticmethod
    def _prev_entry_map(records: list[dict]) -> dict[tuple[str, str], float | None]:
        """Map each (asset, as_of) to the entry price of that asset's PREVIOUS
        forecast, so the momentum baseline can ask "did the last move continue?".
        """
        by_asset: dict[str, list[dict]] = {}
        for r in records:
            by_asset.setdefault(r["asset"], []).append(r)
        out: dict[tuple[str, str], float | None] = {}
        for asset, rs in by_asset.items():
            prev_price: float | None = None
            for r in sorted(rs, key=lambda r: r["as_of"]):
                out[(asset, r["as_of"])] = prev_price
                if r.get("current_price") is not None:
                    prev_price = r["current_price"]
        return out

    @staticmethod
    def _baseline_accuracies(
        recs: list[dict], h: str, prev_entry: dict[tuple[str, str], float | None]
    ) -> dict[str, float | None]:
        """Directional accuracy of three naive predictors at horizon ``h`` — the
        honest floor the desk must beat:

        - ``flat``: always predict Flat (price stays inside the noise band).
        - ``momentum``: the last move continues (persistence / trend-following).
        - ``majority``: the in-sample most common realized label (an optimistic
          oracle reference).

        Realized labels use the per-row deadband that scored the model, so model
        and baselines are compared on identical labels.
        """
        labels: list[str] = []
        flat_correct = 0
        mom_correct = mom_total = 0
        for r in recs:
            entry = r.get("current_price")
            p = r["horizons"][h]
            if entry is None or p.get("realized_price") is None:
                continue
            db = p.get("deadband", DEADBAND)
            label = realized_direction(entry, p["realized_price"], db)
            labels.append(label)
            if label == "Flat":
                flat_correct += 1
            pe = prev_entry.get((r["asset"], r["as_of"]))
            if pe is not None:
                mom_total += 1
                if realized_direction(pe, entry, db) == label:
                    mom_correct += 1
        n = len(labels)
        if n == 0:
            return {"flat": None, "momentum": None, "majority": None}
        majority_count = Counter(labels).most_common(1)[0][1]
        return {
            "flat": flat_correct / n,
            "momentum": mom_correct / mom_total if mom_total else None,
            "majority": majority_count / n,
        }

    def summary(self) -> dict:
        """Aggregate resolved horizons into per-horizon accuracy, calibration, and
        skill-vs-baseline stats."""
        records = self._load()
        prev_entry = self._prev_entry_map(records)
        by_horizon: dict[str, dict] = {}
        for h, _ in FORECAST_HORIZONS:
            recs = [r for r in records
                    if h in r.get("horizons", {})
                    and r["horizons"][h].get("realized_price") is not None]
            rows = [r["horizons"][h] for r in recs]
            dir_rows = [x for x in rows if x.get("direction_correct") is not None]
            rng_rows = [x for x in rows if x.get("in_range") is not None]
            err_rows = [x for x in rows if x.get("abs_pct_error") is not None]

            n_dir = len(dir_rows)
            n_correct = sum(1 for x in dir_rows if x["direction_correct"])
            dir_acc = n_correct / n_dir if n_dir else None

            # Calibration: treat confidence/100 as P(the directional call is correct).
            conf_pairs = [
                (x["confidence"] / 100.0, 1.0 if x["direction_correct"] else 0.0)
                for x in dir_rows if x.get("confidence") is not None
            ]
            brier = (
                sum((p - o) ** 2 for p, o in conf_pairs) / len(conf_pairs)
                if conf_pairs else None
            )
            mean_conf = sum(p for p, _ in conf_pairs) / len(conf_pairs) if conf_pairs else None
            calibration_gap = (
                mean_conf - dir_acc if (mean_conf is not None and dir_acc is not None) else None
            )

            baselines = self._baseline_accuracies(recs, h, prev_entry)
            valid = {k: v for k, v in baselines.items() if v is not None}
            best_name = max(valid, key=lambda k: valid[k]) if valid else None
            best_acc = valid[best_name] if best_name else None
            skill = (
                dir_acc - best_acc if (dir_acc is not None and best_acc is not None) else None
            )

            # Directional commitment: strip out the "easy" Flat predictions and ask
            # whether there is edge when the desk actually picks a side. A row is
            # "committed" when the desk predicted Up/Down; it counts toward
            # right-side accuracy only when the market also actually moved (realized
            # not Flat), so the Flat-dominated base rate can't flatter the number.
            committed = []  # (direction_correct, realized_actually_moved)
            for r in recs:
                p = r["horizons"][h]
                entry = r.get("current_price")
                if entry is None or p.get("direction_correct") is None or p["direction"] == "Flat":
                    continue
                db = p.get("deadband", DEADBAND)
                moved = realized_direction(entry, p["realized_price"], db) != "Flat"
                committed.append((bool(p["direction_correct"]), moved))
            committed_n = len(committed)
            committed_hits = sum(1 for ok, _ in committed if ok)
            side_n = sum(1 for _, moved in committed if moved)
            side_hits = sum(1 for ok, moved in committed if moved and ok)

            by_horizon[h] = {
                "resolved": len(rows),
                "directional_accuracy": dir_acc,
                "directional_accuracy_ci": wilson_interval(n_correct, n_dir) if n_dir else None,
                "sufficient_n": n_dir >= MIN_RESOLVED_N,
                "range_coverage": (
                    sum(1 for x in rng_rows if x["in_range"]) / len(rng_rows)
                    if rng_rows else None
                ),
                "mean_abs_pct_error": (
                    sum(x["abs_pct_error"] for x in err_rows) / len(err_rows)
                    if err_rows else None
                ),
                "brier": brier,
                "mean_confidence": mean_conf,
                "calibration_gap": calibration_gap,
                "reliability": _reliability_bins(conf_pairs),
                "baselines": baselines,
                "best_baseline": best_name,
                "best_baseline_accuracy": best_acc,
                "skill_vs_best_baseline": skill,
                "committed_n": committed_n,
                "committed_rate": committed_n / n_dir if n_dir else None,
                "committed_accuracy": committed_hits / committed_n if committed_n else None,
                "side_n": side_n,
                "side_accuracy": side_hits / side_n if side_n else None,
                "side_accuracy_ci": wilson_interval(side_hits, side_n) if side_n else None,
            }
        return {"total_forecasts": len(records), "by_horizon": by_horizon}

    def summary_markdown(self) -> str:
        s = self.summary()

        def pct(v):
            return "n/a" if v is None else f"{v * 100:.0f}%"

        def signed_pct(v):
            return "n/a" if v is None else f"{v * 100:+.0f}%"

        def acc_cell(st):
            if st["directional_accuracy"] is None:
                return "n/a"
            lo, hi = st["directional_accuracy_ci"]
            flag = "" if st["sufficient_n"] else " ⚠"
            return f"{pct(st['directional_accuracy'])} [{lo * 100:.0f}-{hi * 100:.0f}%]{flag}"

        def base_cell(st):
            if st["best_baseline"] is None:
                return "n/a"
            return f"{pct(st['best_baseline_accuracy'])} ({st['best_baseline']})"

        def brier_cell(st):
            return "n/a" if st["brier"] is None else f"{st['brier']:.3f}"

        def committed_cell(st):
            if st["committed_rate"] is None:
                return "n/a"
            return f"{pct(st['committed_rate'])} ({st['committed_n']})"

        def side_cell(st):
            if st["side_accuracy"] is None:
                return "n/a"
            lo, hi = st["side_accuracy_ci"]
            flag = "" if st["side_n"] >= MIN_RESOLVED_N else " ⚠"
            return f"{pct(st['side_accuracy'])} [{lo * 100:.0f}-{hi * 100:.0f}%]{flag}"

        lines = [
            f"## Forecast track record ({s['total_forecasts']} forecasts logged)",
            "",
            "_Skill = directional acc. − best naive baseline (flat / momentum / "
            "majority): **>0 means the desk beats a no-skill predictor**. Brier "
            "scores the confidence (lower is better; 0.25 = always-50%). "
            f"⚠ = small sample (<{MIN_RESOLVED_N} resolved), read the CI not the point._",
            "",
            "| Horizon | Resolved | Directional acc. (95% CI) | Best baseline | "
            "Skill | Brier | Range coverage | Mean abs % err |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for h, st in s["by_horizon"].items():
            lines.append(
                f"| {h} | {st['resolved']} | {acc_cell(st)} | {base_cell(st)} | "
                f"{signed_pct(st['skill_vs_best_baseline'])} | {brier_cell(st)} | "
                f"{pct(st['range_coverage'])} | {pct(st['mean_abs_pct_error'])} |"
            )

        lines += [
            "",
            "### Directional commitment — is there edge when the desk calls a side?",
            "",
            "_Strips out the easy Flat calls. **Committed** = share of calls where the "
            "desk picked Up/Down (count in parens). **Right-side acc.** is the money "
            "metric: of those committed calls where the market actually moved, how "
            "often the desk picked the right side. **>50% with a CI clear of 50% = "
            "real directional edge**; ≈50% = none._",
            "",
            "| Horizon | Committed | Committed acc. | Right-side acc. (95% CI) |",
            "| --- | ---: | ---: | ---: |",
        ]
        for h, st in s["by_horizon"].items():
            lines.append(
                f"| {h} | {committed_cell(st)} | {pct(st['committed_accuracy'])} | "
                f"{side_cell(st)} |"
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


def forecast_anchor(asset: str, as_of_date: str) -> tuple[str, float] | None:
    """Return ``(as_of_iso, spot_price)`` from the latest bar at/before ``as_of_date``.

    The spot is the real close used as the forecast baseline (and logged as the
    scoring entry), so the displayed anchor and the logged entry always agree.
    Best-effort: returns ``None`` instead of raising.
    """
    try:
        import pandas as pd

        from tradingagents.dataflows.stockstats_utils import load_ohlcv
        df = load_ohlcv(asset, as_of_date)
        if df is None or df.empty:
            return None
        last = df.iloc[-1]
        return pd.to_datetime(last["Date"]).isoformat(), float(last["Close"])
    except Exception:
        return None


def intraday_market_anchor(asset: str, as_of_date: str) -> dict | None:
    """Best-effort spot + realized volatility + recent swing levels at/<= as_of_date.

    Reuses the same intraday bars as ``forecast_anchor`` (so the spot matches the
    logged entry) and adds a per-bar realized volatility (stdev of recent 5m
    percentage returns) and the recent swing high/low. Powers the Portfolio
    Manager's prompt anchor block and the vol-scaled range override. Returns
    ``None`` instead of raising.
    """
    try:
        import pandas as pd

        from tradingagents.dataflows.stockstats_utils import load_ohlcv
        df = load_ohlcv(asset, as_of_date)
        if df is None or df.empty:
            return None
        last = df.iloc[-1]
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        window = close.tail(_SIGMA_WINDOW_BARS)

        sigma_bar = None
        if len(window) >= _SIGMA_MIN_BARS:
            s = float(window.pct_change().dropna().std())
            if s > 0 and not math.isnan(s):
                sigma_bar = s

        return {
            "as_of_iso": pd.to_datetime(last["Date"]).isoformat(),
            "spot": float(last["Close"]),
            "sigma_bar": sigma_bar,
            "recent_low": float(window.min()) if not window.empty else None,
            "recent_high": float(window.max()) if not window.empty else None,
        }
    except Exception:
        return None


def record_forecast(config: dict | None, asset: str, as_of_date: str,
                    forecast_markdown: str, model_version: str = "agents-v1-no-kronos") -> bool:
    """Parse a rendered forecast and log it with the as_of spot price. Never raises."""
    try:
        preds = parse_forecast_markdown(forecast_markdown)
        path = _track_path(config)
        if not preds or not path:
            return False
        anchor = forecast_anchor(asset, as_of_date)
        if anchor is None:
            return False
        as_of_iso, current_price = anchor
        return ForecastTrackRecord(path).log(
            asset, as_of_iso, preds, current_price, model_version
        )
    except Exception:
        return False


def build_price_at() -> Callable[[str, str], float | None]:
    """Realized-price lookup backed by the intraday data layer (per-asset cached)."""
    import pandas as pd

    from tradingagents.dataflows.config import get_config
    from tradingagents.dataflows.stockstats_utils import floor_freq_for, load_ohlcv
    cache: dict[str, object] = {}
    # Floor targets to the base bar size so sub-hourly horizons (5m/15m/30m)
    # resolve against the correct bar — a plain hour floor would misread them.
    freq = floor_freq_for(get_config().get("data_interval", "1h"))

    def price_at(asset: str, target_iso: str) -> float | None:
        try:
            df = cache.get(asset)
            if df is None:
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                d = load_ohlcv(asset, today)
                d = d.assign(_ts=pd.to_datetime(d["Date"])).set_index("_ts").sort_index()
                cache[asset] = d
                df = d
            target = pd.to_datetime(target_iso).floor(freq)
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
    base = (config or {}).get("forecast_deadband_base", DEADBAND)
    tr = ForecastTrackRecord(path)
    tr.resolve(build_price_at(), deadband_base=base)
    return tr.summary_markdown()


def forecast_feedback_block(config: dict | None = None) -> str:
    """Compact per-horizon feedback from the desk's OWN resolved track record, for
    injection into the Portfolio Manager prompt.

    This closes the feedback loop: the PM gets graded against the realized price at
    the RIGHT horizon (5m-4h), so it can calibrate confidence and see where it
    trails a no-skill baseline — instead of the upstream 5-day SPY-alpha "lessons"
    that are meaningless for an intraday call. Returns "" when no horizon has enough
    resolved history yet, so a young/empty log injects nothing. Never raises.
    """
    try:
        path = _track_path(config)
        if not path or not Path(path).expanduser().exists():
            return ""
        summ = ForecastTrackRecord(path).summary()
    except Exception:
        return ""

    lines = []
    for h, st in summ.get("by_horizon", {}).items():
        n = st.get("resolved", 0)
        if n < _FEEDBACK_MIN_RESOLVED or st.get("directional_accuracy") is None:
            continue
        parts = [f"directional {st['directional_accuracy'] * 100:.0f}%"]
        skill, base = st.get("skill_vs_best_baseline"), st.get("best_baseline")
        if skill is not None and base is not None:
            parts.append(f"skill {skill * 100:+.0f}pp vs {base} baseline")
        gap = st.get("calibration_gap")
        if gap is not None:
            if gap > 0.05:
                parts.append(f"confidence ran +{gap * 100:.0f}pp hot — lower it")
            elif gap < -0.05:
                parts.append(f"confidence {-gap * 100:.0f}pp cold — you may raise it")
            else:
                parts.append("confidence well-calibrated")
        cov = st.get("range_coverage")
        if cov is not None:
            parts.append(f"ranges covered {cov * 100:.0f}%")
        side = st.get("side_accuracy")
        if side is not None:
            parts.append(f"right-side {side * 100:.0f}% when committed")
        lines.append(f"- {h}: " + "; ".join(parts) + f" (n={n}).")

    if not lines:
        return ""
    header = (
        "Your own forecasts so far, graded by horizon — calibrate to THESE measured "
        "outcomes, not generic priors (skill <0 means you trailed a no-skill "
        "baseline; trim confidence on horizons where it ran hot):"
    )
    return header + "\n" + "\n".join(lines)
