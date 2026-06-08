import joblib
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from sklearn.linear_model import LogisticRegression

from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """Модель логистической регрессии."""

    def __init__(self, **kwargs):
        super().__init__(name="lr", **kwargs)

        default_params: Dict[str, Any] = {
            "penalty": "l2",           # L2-регуляризация для устойчивости
            "C": 20.0,                  # Сила регуляризации (меньше C → сильнее регуляризация)
            "solver": "lbfgs",
            "max_iter": 10000,          # Максимальное число итераций
            "tol": 1e-3,               # Критерий остановки по изменению коэффициентов
            "class_weight": None,
            "random_state": 42,
            "verbose": 1,
        }
        default_params.update(kwargs)
        self.params = default_params
        self._input_shape = None
        self.threshold_ = 0.59

    def build(self, input_shape: int, **kwargs) -> None:
        """Инициализация sklearn-классификатора."""
        self._input_shape = input_shape
        sk_params = self.params.copy()
        sk_params.pop('input_shape', None)
        self.model = LogisticRegression(**sk_params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            **kwargs) -> None:
        """Обучение модели."""
        if self.model is None:
            self.build(X_train.shape[1])

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        if hasattr(self.model, 'n_iter_'):
            print(f"[LR] Converged in {self.model.n_iter_[0]} iterations.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        proba = self.predict_proba(X)
        return (proba >= self.threshold_).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / "model.joblib"
        data = {
            "model": self.model,
            "params": self.params,
            "input_shape": self._input_shape
        }
        joblib.dump(data, model_path)

    @classmethod
    def load(cls, path: Path) -> "LogisticRegressionModel":
        model_path = path / "model.joblib"
        data = joblib.load(model_path)
        instance = cls(**data["params"])
        instance.model = data["model"]
        instance._input_shape = data.get("input_shape")
        instance.is_fitted = True
        return instance

    def _check_fitted(self):
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")
            