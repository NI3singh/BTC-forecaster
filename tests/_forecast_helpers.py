"""Shared test helper for building forecast fixtures across the suite."""

from tradingagents.agents.schemas import FORECAST_HORIZONS, Direction, Forecast


def make_forecast(direction: Direction = Direction.UP, **overrides):
    """Build a valid six-horizon ``Forecast``.

    Every horizon (5m, 15m, 30m, 1h, 2h, 4h) is filled with sane defaults so all
    required fields are present. ``direction`` sets the default for all horizons;
    override any individual field by name, e.g.::

        make_forecast(direction_4h=Direction.DOWN, expected_price_1h=65950.0)
    """
    fields: dict = {"reasons": "MACD turned up off oversold; ATR widening with horizon."}
    for label, _ in FORECAST_HORIZONS:
        fields[f"direction_{label}"] = direction
        fields[f"expected_price_{label}"] = 65000.0
        fields[f"range_low_{label}"] = 64500.0
        fields[f"range_high_{label}"] = 65500.0
        fields[f"confidence_{label}"] = 55
    fields.update(overrides)
    return Forecast(**fields)
