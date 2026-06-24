"""Triple-barrier labeling for the quant brain (Lopez de Prado, AFML ch. 3).

The fixed-horizon label (``features.make_label``: close[t+h] > close[t]) ignores
the PATH — a move that spikes +2% then reverts scores the same as a steady drift.
The triple-barrier label instead asks which of three barriers price touches FIRST
within the horizon:

  * upper (take-profit) = close_t * (1 + pt * sigma_h)
  * lower (stop-loss)   = close_t * (1 - sl * sigma_h)
  * vertical (time-out) = the bar ``horizon_bars`` ahead

``sigma_h`` is the bar's volatility grown over the horizon (random-walk sqrt-time
scaling), so the barrier means the same thing from 5m to 4h. Touches are detected
on the intrabar HIGH/LOW (not just the close), the way a real stop/limit fills.
The time-out residual is classified with the SAME volatility-scaled dead-band the
live scorer uses (``track_record.deadband_for``), so a label and its later grade
are measured on one yardstick.

LEAKAGE: like ``make_label``, the label for bar t reads bars t+1..t+h — that is
the forward-looking target, by design. Features stay strictly backward-looking.
Rows without a full h-bar future (and the volatility warm-up) come back NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def barrier_sigma(df: pd.DataFrame, window: int = 48) -> pd.Series:
    """Per-bar volatility: rolling std of 5m log returns (backward-looking)."""
    logret = np.log(df["close"] / df["close"].shift(1))
    return logret.rolling(window).std()


def triple_barrier_labels(
    df: pd.DataFrame,
    horizon_bars: int,
    pt: float = 2.0,
    sl: float = 2.0,
    sigma: pd.Series | np.ndarray | None = None,
    sigma_window: int = 48,
    base_deadband: float = 0.001,
) -> pd.DataFrame:
    """First-touch label for each bar over the next ``horizon_bars``.

    Returns a frame indexed like ``df`` with:
      * ``tb_label`` in {-1, 0, +1} (down / flat-timeout / up); NaN where the
        future window or the volatility warm-up is incomplete,
      * ``tb_ret`` signed realized log-return of the outcome (to the touched
        barrier, or to the time-out close),
      * ``touch_bar`` bars until first touch (``horizon_bars`` on time-out).

    ``sigma`` (per-bar vol) may be passed in to reuse one estimate across
    horizons / make tests deterministic; otherwise ``barrier_sigma`` computes it.
    """
    from tradingagents.forecasting.track_record import deadband_for

    n = len(df)
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    if sigma is None:
        sigma = barrier_sigma(df, sigma_window)
    sigma = np.asarray(sigma, dtype=float)

    sigma_h = sigma * np.sqrt(horizon_bars)
    upper = close * (1.0 + pt * sigma_h)
    lower = close * (1.0 - sl * sigma_h)

    label = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    touch = np.full(n, float(horizon_bars))
    hit = np.zeros(n, dtype=bool)

    # Only bars with a full future window AND a defined sigma can be labeled.
    valid = np.zeros(n, dtype=bool)
    if n > horizon_bars:
        valid[: n - horizon_bars] = True
    valid &= ~np.isnan(sigma_h)

    for k in range(1, horizon_bars + 1):
        fh = np.full(n, -np.inf)
        fl = np.full(n, np.inf)
        fh[: n - k] = high[k:]
        fl[: n - k] = low[k:]
        live = valid & ~hit
        up_hit = live & (fh >= upper)
        dn_hit = live & (fl <= lower)
        # Same-bar double touch resolves DOWN (conservative; rare in practice).
        up_only = up_hit & ~dn_hit
        label[dn_hit] = -1.0
        touch[dn_hit] = k
        ret[dn_hit] = np.log(lower[dn_hit] / close[dn_hit])
        hit[dn_hit] = True
        label[up_only] = 1.0
        touch[up_only] = k
        ret[up_only] = np.log(upper[up_only] / close[up_only])
        hit[up_only] = True
        if not (valid & ~hit).any():
            break

    # Time-outs: never touched a horizontal barrier within the window. Classify
    # the close-to-close residual with the live (volatility-scaled) dead-band.
    timeout = valid & ~hit
    if timeout.any():
        fc = np.full(n, np.nan)
        fc[: n - horizon_bars] = close[horizon_bars:]
        db = deadband_for(horizon_bars * 5, base=base_deadband)
        net = np.log(fc / close)
        ret[timeout] = net[timeout]
        label[timeout] = 0.0
        label[timeout & (net > np.log1p(db))] = 1.0
        label[timeout & (net < np.log1p(-db))] = -1.0

    out = pd.DataFrame(
        {"tb_label": label, "tb_ret": ret, "touch_bar": touch}, index=df.index
    )
    out.loc[~valid, ["tb_label", "tb_ret", "touch_bar"]] = np.nan
    return out
