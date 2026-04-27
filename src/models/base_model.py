"""
Абстрактный базовый класс с унифицированным интерфейсом и подбором порога.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_curve


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

    def tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        probas = self.predict_proba(X_val)
        prec, rec, thresh = precision_recall_curve(y_val, probas)
        thresh = np.append(thresh, 1.0)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-10)
        best = np.argmax(f1s)
        self.threshold_ = thresh[best]
        return self.threshold_

    @abstractmethod
    def save(self, path: Path) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        pass
