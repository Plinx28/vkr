"""
Реализация логистической регрессии через sklearn.
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Optional
from sklearn.linear_model import LogisticRegression

from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(name="logistic_regression", **kwargs)
        # Параметры по умолчанию
        default_params = {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "class_weight": "balanced",  # Учитываем дисбаланс классов
            "random_state": 42,
        }
        default_params.update(kwargs)
        self.params = default_params

    def build(self, input_shape: int, **kwargs) -> None:
        self.params["input_shape"] = input_shape
        self.model = LogisticRegression(**self.params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            **kwargs) -> None:
        if self.model is None:
            self.build(X_train.shape[1])
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        # Возвращаем вероятности класса 1 (аномалия)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / "model.joblib"
        joblib.dump({"model": self.model, "params": self.params}, model_path)

    @classmethod
    def load(cls, path: Path) -> "LogisticRegressionModel":
        model_path = path / "model.joblib"
        data = joblib.load(model_path)
        instance = cls(**data["params"])
        instance.model = data["model"]
        instance.is_fitted = True
        return instance

    def _check_fitted(self):
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")