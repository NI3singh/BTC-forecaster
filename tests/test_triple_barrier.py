"""Tests for triple-barrier labeling (hand-built frames, deterministic sigma)."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.forecasting.quant.triple_barrier import (
    barrier_sigma,
    triple_barrier_labels,
)
from tradingagents.forecasting.track_record import deadband_for


def _frame(close, high, low):
    n = len(close)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": list(close), "high": list(high), "low": list(low), "close": list(close)},
        index=idx,
    )


_SIG = 0.01  # constant per-bar sigma -> sigma_h=0.02 at h=4 -> barriers at +/-2%


@pytest.mark.unit
class TestTripleBarrier:
    def test_upper_touch_first(self):
        close = [100.0] * 8
        high = [100, 101, 105, 100, 100, 100, 100, 100]
        low = [100, 99, 99, 99, 99, 100, 100, 100]
        out = triple_barrier_labels(_frame(close, high, low), 4, pt=2, sl=2,
                                    sigma=np.full(8, _SIG))
        assert out["tb_label"].iloc[0] == 1.0          # upper (104) hit at k=2
        assert out["touch_bar"].iloc[0] == 2
        assert out["tb_ret"].iloc[0] == pytest.approx(np.log(104 / 100))

    def test_lower_touch_first(self):
        close = [100.0] * 8
        high = [100, 101, 101, 100, 100, 100, 100, 100]
        low = [100, 99, 95, 99, 99, 100, 100, 100]
        out = triple_barrier_labels(_frame(close, high, low), 4, pt=2, sl=2,
                                    sigma=np.full(8, _SIG))
        assert out["tb_label"].iloc[0] == -1.0         # lower (96) hit at k=2
        assert out["touch_bar"].iloc[0] == 2
        assert out["tb_ret"].iloc[0] == pytest.approx(np.log(96 / 100))

    def test_timeout_is_flat(self):
        close = [100.0] * 8
        high = [100, 101, 102, 103, 102, 101, 100, 100]   # max 103 < 104
        low = [100, 99, 98, 97, 98, 99, 100, 100]         # min 97 > 96
        out = triple_barrier_labels(_frame(close, high, low), 4, pt=2, sl=2,
                                    sigma=np.full(8, _SIG))
        assert out["tb_label"].iloc[0] == 0.0            # never touched, no drift
        assert out["touch_bar"].iloc[0] == 4
        assert out["tb_ret"].iloc[0] == pytest.approx(0.0)

    def test_same_bar_double_touch_resolves_down(self):
        close = [100.0] * 8
        high = [100, 105, 100, 100, 100, 100, 100, 100]   # breaches upper at k=1
        low = [100, 95, 100, 100, 100, 100, 100, 100]     # AND lower at k=1
        out = triple_barrier_labels(_frame(close, high, low), 4, pt=2, sl=2,
                                    sigma=np.full(8, _SIG))
        assert out["tb_label"].iloc[0] == -1.0           # tie -> downside (conservative)
        assert out["touch_bar"].iloc[0] == 1

    def test_timeout_residual_uses_deadband(self):
        # No barrier touch; a close-to-close drift ABOVE deadband_for -> Up (not Flat).
        h = 4
        db = deadband_for(h * 5, base=0.001)
        drifted = 100 * np.exp(np.log1p(db) + 0.001)      # just past the band, < upper
        close = [100.0, 100, 100, 100, drifted, 100, 100, 100]
        high = [100, 100.5, 101, 101, drifted, 100, 100, 100]
        low = [100, 99.5, 99, 99, 99, 100, 100, 100]
        out = triple_barrier_labels(_frame(close, high, low), h, pt=2, sl=2,
                                    sigma=np.full(8, _SIG))
        assert out["tb_label"].iloc[0] == 1.0
        assert out["touch_bar"].iloc[0] == h             # timed out (no barrier touch)

    def test_label_ignores_bars_beyond_horizon(self):
        close = [100.0] * 12
        high = [100, 101, 102, 103, 102, 200, 100, 100, 100, 100, 100, 100]
        low = [100, 99, 98, 97, 98, 99, 100, 100, 100, 100, 100, 100]
        # bar 0 (h=4) sees bars 1..4 only; the bar-5 spike to 200 must not leak in.
        out = triple_barrier_labels(_frame(close, high, low), 4, pt=2, sl=2,
                                    sigma=np.full(12, _SIG))
        assert out["tb_label"].iloc[0] == 0.0

    def test_tail_rows_are_nan(self):
        flat = [100.0] * 10
        out = triple_barrier_labels(_frame(flat, flat, flat), 4, sigma=np.full(10, _SIG))
        assert out["tb_label"].iloc[-4:].isna().all()    # no full future window

    def test_sigma_has_no_lookahead(self):
        rng = np.random.default_rng(0)
        close = 100 + rng.normal(0, 1, 60).cumsum()
        df = _frame(close, close + 0.5, close - 0.5)
        s1 = barrier_sigma(df, window=5)
        df2 = df.copy()
        df2.iloc[-1, df2.columns.get_loc("close")] += 100.0   # perturb the LAST bar
        s2 = barrier_sigma(df2, window=5)
        assert s1.iloc[20] == pytest.approx(s2.iloc[20])      # early bar unaffected
