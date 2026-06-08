"""
Абстрактный базовый класс с унифицированным интерфейсом и подбором порога.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef


class BaseModel(ABC):
    """Абстрактный базовый класс для моделей."""
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.model: Any = None
        self.params = kwargs
        self.is_fitted = False
        self.threshold_ = 0.5

    @abstractmethod
    def build(self, input_shape: int, **kwargs) -> None:
        pass

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            **kwargs) -> Any:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        pass

    def optimize_threshold(self, X_val: np.ndarray, y_val: np.ndarray, metric: str = "f1") -> float:
        """Подбирает порог, максимизирующий F1 или MCC на валидации."""
        probas = self.predict_proba(X_val)
        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = []
        for thr in thresholds:
            preds = (probas >= thr).astype(int)
            if metric == "f1":
                scores.append(f1_score(y_val, preds))
            elif metric == "mcc":
                scores.append(matthews_corrcoef(y_val, preds))
        best_idx = np.argmax(scores)
        self.threshold_ = thresholds[best_idx]
        return self.threshold_
