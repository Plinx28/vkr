"""
Логистическая регрессия с регуляризацией и автоматическим взвешиванием классов.
Использует solver 'saga' для поддержки больших данных и быстрой сходимости.
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from sklearn.linear_model import LogisticRegression

from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """
    Модель логистической регрессии с предустановленными параметрами,
    оптимизированными для задачи бинарной классификации сетевого трафика.
    """

    def __init__(self, **kwargs):
        super().__init__(name="logistic_regression", **kwargs)

        # Параметры по умолчанию, подобранные для:
        # - дисбаланса классов (class_weight='balanced')
        # - предотвращения переобучения (регуляризация L2 с C=1.0)
        # - быстрой сходимости на больших данных (solver='saga', tol=1e-4)
        default_params: Dict[str, Any] = {
            "penalty": "l2",           # L2-регуляризация для устойчивости
            "C": 1.0,                  # Сила регуляризации (меньше C → сильнее регуляризация)
            "solver": "saga",          # Поддерживает L1/L2, хорошо работает на больших выборках
            "max_iter": 1000,          # Максимальное число итераций
            "tol": 1e-4,               # Критерий остановки по изменению коэффициентов
            "class_weight": "balanced", # Автоматическое взвешивание классов
            "random_state": 42,
            "n_jobs": -1,              # Использовать все ядра CPU
            "verbose": 0,
        }
        default_params.update(kwargs)
        self.params = default_params
        self._input_shape = None

    def build(self, input_shape: int, **kwargs) -> None:
        """Инициализация sklearn-классификатора."""
        self._input_shape = input_shape
        # Создаём копию параметров без 'input_shape'
        sk_params = self.params.copy()
        sk_params.pop('input_shape', None)
        self.model = LogisticRegression(**sk_params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            **kwargs) -> None:
        """
        Обучение модели.
        Для логистической регрессии валидационная выборка не используется,
        так как встроенная регуляризация и tol предотвращают переобучение.
        """
        if self.model is None:
            self.build(X_train.shape[1])

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Логируем число итераций, если доступно
        if hasattr(self.model, 'n_iter_'):
            print(f"[LR] Converged in {self.model.n_iter_[0]} iterations.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / "model.joblib"
        # Сохраняем также input_shape для информации
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
            