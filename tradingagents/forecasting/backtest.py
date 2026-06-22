"""Offline, price-only baseline backtest for the forecast horizons.

The forecast track record can only grow one elapsed horizon at a time, so judging
a change to the forecaster (or just asking "do we even beat a coin flip?") means
waiting hours of live data. This module replays historical bars instead: at a grid
of past timestamps it computes what the naive predictors (flat / momentum /
majority) would have scored at each horizon — the honest performance floor — in
seconds and with no agent/LLM calls.

The core (``backtest_baselines``) is pure: it takes a bar frame and is fully
network-free and testable. ``run_backtest`` is the thin wiring that backs it with
the live data layer (``load_ohlcv``, which already filters look-ahead).
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from tradingagents.agents.schemas import FORECAST_HORIZONS
from tradingagents.forecasting.track_record import (
    DEADBAND,
    deadband_for,
    realized_direction,
)


def backtest_baselines(
    bars: pd.DataFrame,
    *,
    step: int = 1,
    deadband_base: float = DEADBAND,
) -> dict[str, dict]:
    """Score the naive baselines at every horizon over ``bars``.

    ``bars`` must be a DataFrame with a sorted ``DatetimeIndex`` and a ``Close``
    column. From each anchor bar (taking every ``step``-th bar) the entry close is
    compared, per horizon, against the realized close once that horizon has
    elapsed. Realized labels use the same ``deadband_for`` band the live scorer
    uses, so backtest and live numbers are directly comparable.

    Returns ``{horizon: {n, flat_acc, momentum_acc, majority_acc, majority_label}}``.
    Lookups are O(log n) via ``searchsorted``, so a full 60-day 5m history scores
    in well under a second.
    """
    idx = bars.index
    close = bars["Close"]
    n = len(bars)

    acc: dict[str, dict] = {
        h: {"n": 0, "flat": 0, "momentum": 0, "mom_total": 0, "labels": []}
        for h, _ in FORECAST_HORIZONS
    }

    for i in range(0, n, step):
        ts = idx[i]
        entry = float(close.iloc[i])
        for h, minutes in FORECAST_HORIZONS:
            db = deadband_for(minutes, base=deadband_base)
            delta = pd.Timedelta(minutes=minutes)

            # Realized close at the horizon: the last bar at/before (ts + h),
            # requiring a strictly later bar to exist so the horizon has elapsed.
            p = idx.searchsorted(ts + delta, side="right")
            if p < 1 or p >= n:
                continue
            realized = float(close.iloc[p - 1])
            label = realized_direction(entry, realized, db)

            rec = acc[h]
            rec["n"] += 1
            rec["labels"].append(label)
            if label == "Flat":
                rec["flat"] += 1

            # Momentum / persistence: the move over the prior h-window continues.
            q = idx.searchsorted(ts - delta, side="right")
            if q >= 1:
                past = float(close.iloc[q - 1])
                rec["mom_total"] += 1
                if realized_direction(past, entry, db) == label:
                    rec["momentum"] += 1

    out: dict[str, dict] = {}
    for h, _ in FORECAST_HORIZONS:
        rec = acc[h]
        n_h = rec["n"]
        if n_h == 0:
            out[h] = {"n": 0}
            continue
        majority_label, majority_count = Counter(rec["labels"]).most_common(1)[0]
        out[h] = {
            "n": n_h,
            "flat_acc": rec["flat"] / n_h,
            "momentum_acc": rec["momentum"] / rec["mom_total"] if rec["mom_total"] else None,
            "majority_acc": majority_count / n_h,
            "majority_label": majority_label,
        }
    return out


def run_backtest(
    asset: str, config: dict | None = None, *, step: int = 12
) -> dict[str, dict]:
    """Back the pure baseline backtest with the live intraday data layer.

    Loads cached/refetched bars via ``load_ohlcv`` (look-ahead already filtered)
    and scores the baselines. ``step`` is in bars: 12 on 5m data anchors hourly,
    keeping the run cheap while still sampling the whole window. Best-effort —
    returns ``{}`` if data cannot be loaded.
    """
    from datetime import datetime, timezone

    from tradingagents.dataflows.config import get_config
    from tradingagents.dataflows.stockstats_utils import load_ohlcv

    cfg = config or get_config()
    base = cfg.get("forecast_deadband_base", DEADBAND)
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        df = load_ohlcv(asset, today)
        if df is None or df.empty:
            return {}
        bars = df.assign(_ts=pd.to_datetime(df["Date"])).set_index("_ts").sort_index()
        return backtest_baselines(bars, step=step, deadband_base=base)
    except Exception:
        return {}


def backtest_markdown(asset: str, results: dict[str, dict]) -> str:
    """Render a baseline-backtest result table as markdown."""

    def pct(v):
        return "n/a" if v is None else f"{v * 100:.0f}%"

    if not results:
        return f"## Baseline backtest — {asset}\n\nNo historical bars available."

    lines = [
        f"## Baseline backtest — {asset} (price-only, no agents)",
        "",
        "_The floor the agent forecaster must beat at each horizon. flat = always "
        "Flat; momentum = the prior h-window move continues; majority = the most "
        "common realized label. If the desk's live directional accuracy is not "
        "clearly above these, it has no measured edge._",
        "",
        "| Horizon | Samples | Flat | Momentum | Majority (label) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for h, _ in FORECAST_HORIZONS:
        st = results.get(h, {})
        n = st.get("n", 0)
        if not n:
            lines.append(f"| {h} | 0 | n/a | n/a | n/a |")
            continue
        majority = f"{pct(st.get('majority_acc'))} ({st.get('majority_label')})"
        lines.append(
            f"| {h} | {n} | {pct(st.get('flat_acc'))} | "
            f"{pct(st.get('momentum_acc'))} | {majority} |"
        )
    return "\n".join(lines)
