"""Quantitative forecasting brain: gradient-boosted per-horizon direction models.

Adapted from the sibling ``trading_bot`` project (an honest, walk-forward BTC
direction predictor) and repurposed for this multi-horizon intraday forecaster:
5-minute Binance bars, one HistGradientBoosting model per horizon (5m..4h), each
producing a calibrated ``P(up)``. The point is to replace the Portfolio Manager's
free-hand directional guess with a measured, baseline-checked prior — and to be
honest, per horizon, about where there is and isn't an edge.
"""

from tradingagents.forecasting.quant.forecaster import QuantForecaster

__all__ = ["QuantForecaster"]
