# Vendored Kronos model code

This directory contains the inference code for **Kronos** (the `model/` package),
vendored from the upstream repository so BTC-Forecaster can run Kronos zero-shot
without an external clone on the import path.

- **Upstream:** https://github.com/shiyu-coder/Kronos
- **Pinned commit:** `67b630e67f6a18c9e9be918d9b4337c960db1e9a`
- **License:** MIT (see `LICENSE` in this directory — Copyright (c) 2025 ShiYu)
- **Files:** `kronos.py`, `module.py`, `__init__.py` (copied verbatim except the edits below)

## Local modifications (`kronos.py` only)

1. `from model.module import *` → `from .module import *` (and removed the
   `sys.path.append("../")` hack) so the package imports cleanly when vendored.
2. Added a `return_samples` flag to `auto_regressive_inference` / `KronosPredictor.generate`
   and a new `KronosPredictor.predict_samples(...)` method that returns the **per-sample**
   forecast paths instead of their mean — the public `predict()` averages the samples and
   discards the spread we need to build a directional probability / range distribution.

`module.py` and `__init__.py` are unmodified.

Model weights are **not** vendored; they download from the Hugging Face Hub
(`NeoQuasar/Kronos-small`, `NeoQuasar/Kronos-Tokenizer-base`) on first use.
