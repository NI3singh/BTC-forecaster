"""Kronos: a zero-shot generative quant brain (optional; needs the ``[kronos]`` extra
to actually run a forecast — the package itself imports without ``torch``)."""

from tradingagents.forecasting.kronos.forecaster import (
    KronosForecaster,
    aggregate_paths,
    kronos_eval_markdown,
    render_kronos_block,
)

__all__ = [
    "KronosForecaster",
    "aggregate_paths",
    "kronos_eval_markdown",
    "render_kronos_block",
]
