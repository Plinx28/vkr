"""
Реализация градиентного бустинга XGBoost.
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Optional
import xgboost as xgb

from .base_model import BaseModel


class XGBoostModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(name="xgboost", **kwargs)
        # Параметры по умолчанию, оптимизированные для дисбаланса
        default_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": None,  # Будет вычислен автоматически при fit
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "verbosity": 0,
        }
        default_params.update(kwargs)
        self.params = default_params

    def build(self, input_shape: int, **kwargs) -> None:
        self.params["input_shape"] = input_shape
        self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            **kwargs) -> None:
        if self.model is None:
            self.build(X_train.shape[1])

        # Автоматическое вычисление scale_pos_weight, если не задан явно
        if self.params.get("scale_pos_weight") is None:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale = neg_count / pos_count if pos_count > 0 else 1.0
            self.model.set_params(scale_pos_weight=scale)

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=kwargs.get("verbose", False)
        )
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / "model.json"
        self.model.save_model(model_path)
        # Сохраняем параметры отдельно
        params_path = path / "params.joblib"
        joblib.dump(self.params, params_path)

    @classmethod
    def load(cls, path: Path) -> "XGBoostModel":
        model_path = path / "model.json"
        params_path = path / "params.joblib"
        params = joblib.load(params_path)
        instance = cls(**params)
        instance.model = xgb.XGBClassifier()
        instance.model.load_model(model_path)
        instance.is_fitted = True
        return instance

    def _check_fitted(self):
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")