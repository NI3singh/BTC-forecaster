"""Per-horizon gradient-boosted direction model + honest walk-forward evaluation.

Adapted from trading_bot/src/model.py and baselines.py. Each horizon gets its own
HistGradientBoostingClassifier producing ``P(up)``. Evaluation is expanding-window
walk-forward (train on the past, test on the future, never shuffled) scored
against the same baselines a real edge must beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def make_model() -> HistGradientBoostingClassifier:
    """Gradient-boosted trees with conservative anti-overfit settings."""
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.03,
        max_depth=4,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )


def _usable_columns(X: pd.DataFrame) -> list[str]:
    """Columns HistGradientBoosting can bin: drop all-NaN or constant features.

    Its binning step fails on a feature with <2 distinct non-NaN values — which an
    optional derivative feature can be in an early walk-forward fold that predates
    the ~30-day futures-data window. Dropping such columns per-fit keeps training
    robust; a feature simply isn't used where it carries no information.
    """
    return [c for c in X.columns if X[c].nunique(dropna=True) > 1]


# --- Baselines the model must beat (computed on the SAME out-of-sample rows) ---

def baseline_always_up(n: int) -> np.ndarray:
    return np.ones(n)


def baseline_majority(y_train: pd.Series, n: int) -> np.ndarray:
    return np.full(n, 1.0 if y_train.mean() >= 0.5 else 0.0)


def baseline_persistence(df: pd.DataFrame, idx: pd.Index) -> np.ndarray:
    """Momentum: predict the next move equals the last completed move."""
    last_move = (df["close"] > df["close"].shift(1)).astype(float)
    return last_move.reindex(idx).fillna(1.0).to_numpy()


def generate_oos_predictions(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    """Expanding-window walk-forward predictions covering each OOS row once.

    Returns a frame indexed by timestamp with columns [prob_up, y_true].
    """
    n = len(X)
    fold = n // (n_splits + 1)
    frames = []
    for k in range(1, n_splits + 1):
        train_end = fold * k
        test_end = fold * (k + 1) if k < n_splits else n
        Xtr, ytr = X.iloc[:train_end], y.iloc[:train_end]
        Xte, yte = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
        if len(Xte) == 0:
            continue
        model = make_model()
        cols = _usable_columns(Xtr)
        model.fit(Xtr[cols], ytr)
        prob = model.predict_proba(Xte[cols])[:, 1]
        frames.append(pd.DataFrame({"prob_up": prob, "y_true": yte.to_numpy()}, index=Xte.index))
    return pd.concat(frames) if frames else pd.DataFrame(columns=["prob_up", "y_true"])


def evaluate_horizon(X: pd.DataFrame, y: pd.Series, df: pd.DataFrame,
                     n_splits: int = 5) -> dict:
    """Walk-forward OOS accuracy + AUC vs baselines for one horizon."""
    fold = len(X) // (n_splits + 1)
    preds = generate_oos_predictions(X, y, n_splits=n_splits)
    if preds.empty:
        return {"n_test": 0}
    y_true = preds["y_true"].to_numpy()
    y_prob = preds["prob_up"].to_numpy()
    y_pred = (y_prob >= 0.5).astype(float)

    base = {
        "always_up": accuracy_score(y_true, baseline_always_up(len(y_true))),
        "majority": accuracy_score(y_true, baseline_majority(y.iloc[:fold], len(y_true))),
        "persistence": accuracy_score(y_true, baseline_persistence(df, preds.index)),
    }
    model_acc = accuracy_score(y_true, y_pred)
    # AUC is undefined if the OOS slice happens to be one-class.
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return {
        "model_acc": model_acc,
        "model_auc": auc,
        "baselines": base,
        "best_baseline": max(base, key=lambda k: base[k]),
        "best_baseline_acc": max(base.values()),
        "edge": model_acc - max(base.values()),
        "n_test": len(y_true),
        "up_rate": float(y_true.mean()),
    }


def train_full(X: pd.DataFrame, y: pd.Series) -> HistGradientBoostingClassifier:
    """Fit one model on all available rows (for live prediction)."""
    model = make_model()
    cols = _usable_columns(X)
    model.fit(X[cols], y)
    return model
