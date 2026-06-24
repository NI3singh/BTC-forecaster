"""Deterministic, volatility-scaled price ranges for the forecast horizons.

The Portfolio Manager LLM is decent at calling direction and a rough expected
price, but its hand-typed ranges are unprincipled — frequently too narrow, not
guaranteed to widen with the horizon, occasionally inverted. This module replaces
the WIDTH of each horizon's range with a sqrt-time-scaled volatility band centered
on the LLM's own expected price, so coverage becomes a controllable,
monotonically-widening quantity while the desk's directional view is preserved.

Pure and network-free: ``apply_vol_scaled_ranges`` transforms a typed Forecast and
``render_anchor_block`` formats prompt text; the volatility/price inputs are
supplied by the caller (the Portfolio Manager node, from the live data layer).
"""

from __future__ import annotations

import math

from tradingagents.agents.schemas import FORECAST_HORIZONS, Forecast

# Band half-width = RANGE_K * sigma_bar * sqrt(horizon_bars). ~1.5 is near an 80%
# interval for roughly-normal 5m returns; the eventual conformal calibration
# (Tier 1) tunes this per horizon from realized coverage.
RANGE_K = 1.5

# The base OHLCV bar is 5 minutes, so horizon_bars = horizon_minutes / 5.
_BASE_BAR_MINUTES = 5


def vol_scaled_band(
    expected: float, sigma_bar: float, minutes: int, k: float = RANGE_K
) -> tuple[float, float]:
    """``(low, high)`` around ``expected`` for a horizon ``minutes`` ahead.

    ``sigma_bar`` is the per-base-bar return volatility as a fraction of price; the
    half-width grows with the square root of time: ``k * sigma_bar * sqrt(bars)``.
    """
    bars = max(minutes / _BASE_BAR_MINUTES, 0.0)
    half_width = k * sigma_bar * math.sqrt(bars)
    return (expected * (1 - half_width), expected * (1 + half_width))


def apply_vol_scaled_ranges(
    forecast: Forecast, sigma_bar: float, k: float = RANGE_K
) -> Forecast:
    """Recompute every horizon's ``range_low``/``range_high`` as a vol-scaled band
    around that horizon's expected price, in place, and return ``forecast``.

    Direction, expected_price, and confidence are untouched — only the band WIDTH
    becomes principled. Bands widen monotonically with the horizon by construction
    and always satisfy ``low <= expected <= high``. A missing/non-positive/NaN
    ``sigma_bar`` leaves the forecast unchanged (best-effort, never raises).
    """
    if not sigma_bar or sigma_bar <= 0 or math.isnan(sigma_bar):
        return forecast
    for label, minutes in FORECAST_HORIZONS:
        expected = getattr(forecast, f"expected_price_{label}")
        low, high = vol_scaled_band(expected, sigma_bar, minutes, k)
        setattr(forecast, f"range_low_{label}", low)
        setattr(forecast, f"range_high_{label}", high)
    return forecast


def apply_interval_ranges(
    forecast: Forecast, offsets: dict[str, tuple[float, float]]
) -> Forecast:
    """Override ``range_low``/``range_high`` from conditional interval offsets.

    ``offsets`` maps a horizon to ``(lo_frac, hi_frac)`` fractional bounds around the
    expected price (from the volatility brain's HAR+conformal interval). Horizons
    absent from ``offsets`` keep whatever ``apply_vol_scaled_ranges`` set (so this is
    a best-effort sharpening layer on top of the constant-σ band). In place.
    """
    for label, _ in FORECAST_HORIZONS:
        off = offsets.get(label)
        if not off:
            continue
        lo_frac, hi_frac = off
        expected = getattr(forecast, f"expected_price_{label}")
        if expected is None:
            continue
        setattr(forecast, f"range_low_{label}", expected * (1.0 + lo_frac))
        setattr(forecast, f"range_high_{label}", expected * (1.0 + hi_frac))
    return forecast


def render_anchor_block(asset: str, anchor: dict) -> str:
    """Compact 'source of truth' block for the PM prompt: the real spot price and
    the realized volatility scale (plus recent swing levels), so the model anchors
    expected prices on a true value instead of numbers paraphrased through debate.
    """
    spot = anchor["spot"]
    lines = [
        f"**ANCHOR — verified source of truth for {asset} (use these exact numbers; "
        "do NOT invent the current price or volatility):**",
        f"- Spot price now: ${spot:,.2f}",
    ]
    sigma = anchor.get("sigma_bar")
    if sigma and sigma > 0:
        one_h = sigma * math.sqrt(60 / _BASE_BAR_MINUTES) * 100
        four_h = sigma * math.sqrt(240 / _BASE_BAR_MINUTES) * 100
        lines.append(
            f"- Realized volatility: ~{sigma * 100:.2f}% per 5m bar "
            f"(≈ ±{one_h:.2f}% over 1h, ±{four_h:.2f}% over 4h). "
            "Keep expected prices within this scale."
        )
    low, high = anchor.get("recent_low"), anchor.get("recent_high")
    if low and high:
        lines.append(f"- Recent range: ${low:,.2f} (low) – ${high:,.2f} (high)")
    return "\n".join(lines)
