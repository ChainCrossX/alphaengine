"""
Directional ML model.

Honest note: short-horizon direction prediction on liquid US equities is mostly noise.
This model is here as a *minor confirming signal*, not as an alpha source. We use a
regularized gradient boosted tree (LightGBM) on engineered features with walk-forward
validation. If the OOS log-loss does not beat a 0.5 baseline, the model returns flat
probabilities and contributes zero to the ensemble.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _LGB = True
except ImportError:
    _LGB = False

from .indicators import feature_frame
from .logger import get_logger

log = get_logger(__name__)


FEATURES = [
    "ret_1", "ret_5", "ret_20",
    "rsi_14", "macd", "signal", "hist",
    "adx", "+di", "-di",
    "bb_width", "bb_pct_b",
    "atr_14", "roc_12",
    "rv_20", "volume_z",
]


@dataclass
class TrainResult:
    valid_logloss: float
    baseline_logloss: float = 0.6931  # -log(0.5)
    is_useful: bool = False


class DirectionModel:
    """
    Predicts P(next bar return > 0). Trained per-symbol or pooled.

    `predict_last(df)` returns (p_up, p_down) for the most recent feature row.
    """

    def __init__(self, horizon: int = 1, max_depth: int = 5, n_estimators: int = 300):
        if not _LGB:
            raise ImportError("LightGBM is required for DirectionModel")
        self.horizon = horizon
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.model = None
        self.train_result: TrainResult | None = None

    @staticmethod
    def _make_xy(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
        feats = feature_frame(df)
        target = (feats["close"].shift(-horizon) > feats["close"]).astype(int)
        feats = feats.dropna()
        target = target.loc[feats.index].dropna()
        feats = feats.loc[target.index]
        X = feats[FEATURES]
        return X, target

    def fit(self, df: pd.DataFrame, valid_frac: float = 0.2) -> TrainResult:
        X, y = self._make_xy(df, self.horizon)
        if len(X) < 200:
            log.warning("Not enough data to train ML model")
            self.model = None
            self.train_result = TrainResult(valid_logloss=1.0, is_useful=False)
            return self.train_result

        cut = int(len(X) * (1 - valid_frac))
        X_tr, X_va = X.iloc[:cut], X.iloc[cut:]
        y_tr, y_va = y.iloc[:cut], y.iloc[cut:]

        model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_samples=30,
            random_state=42,
            verbose=-1,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        proba = model.predict_proba(X_va)[:, 1]
        eps = 1e-9
        ll = -float(np.mean(y_va * np.log(proba + eps) + (1 - y_va) * np.log(1 - proba + eps)))
        useful = ll < 0.6931 - 0.005  # require meaningful improvement vs random
        self.model = model if useful else None
        self.train_result = TrainResult(valid_logloss=ll, is_useful=useful)
        log.info(
            f"ML train: logloss={ll:.4f} useful={useful} "
            f"(baseline=0.6931)"
        )
        return self.train_result

    def predict_last(self, df: pd.DataFrame) -> tuple[float, float]:
        if self.model is None:
            return 0.5, 0.5
        feats = feature_frame(df).dropna()
        if feats.empty:
            return 0.5, 0.5
        x = feats[FEATURES].iloc[[-1]]
        p_up = float(self.model.predict_proba(x)[0, 1])
        return p_up, 1 - p_up

    def save(self, path: str | Path) -> None:
        if self.model is None:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(path))

    def load(self, path: str | Path) -> None:
        booster = lgb.Booster(model_file=str(path))
        wrapper = lgb.LGBMClassifier()
        wrapper._Booster = booster
        wrapper._n_classes = 2
        self.model = wrapper
