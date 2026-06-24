"""Lazy, cached loading of the Kronos tokenizer + model + predictor.

Weights download from the HuggingFace Hub on first use (``NeoQuasar/Kronos-small`` +
``NeoQuasar/Kronos-Tokenizer-base``). The predictor is built once per
``(model, tokenizer, device, max_context)`` and reused across forecasts so a live
run stays cheap after the first load. ``torch`` and the vendored model code are
imported lazily here, so the package imports fine without the optional ``[kronos]``
extra installed — you only need it when you actually run a Kronos forecast.
"""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
_PREDICTOR = None
_KEY: tuple | None = None


def get_predictor(model_id: str, tokenizer_id: str, device: str, max_context: int):
    """Return a cached ``KronosPredictor``, loading the model/tokenizer on first call."""
    global _PREDICTOR, _KEY
    key = (model_id, tokenizer_id, device, max_context)
    with _LOCK:
        if _PREDICTOR is not None and key == _KEY:
            return _PREDICTOR
        from tradingagents.forecasting.kronos._vendor import (
            Kronos,
            KronosPredictor,
            KronosTokenizer,
        )

        tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
        model = Kronos.from_pretrained(model_id)
        model.eval()
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_context)
        _PREDICTOR, _KEY = predictor, key
        return predictor
