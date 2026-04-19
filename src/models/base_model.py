"""
Абстрактный базовый класс для всех моделей.
Определяет унифицированный интерфейс fit, predict, predict_proba, save, load.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
import numpy as np
from pathlib import Path


class BaseModel(ABC):
    """Абстрактный класс модели."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.model: Any = None
        self.params = kwargs
        self.is_fitted = False

    @abstractmethod
    def build(self, input_shape: int, **kwargs) -> None:
        """
        Создание архитектуры модели (для нейросетей) или инициализация объекта.
        Вызывается перед обучением.
        """
        pass

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            **kwargs) -> Any:
        """
        Обучение модели. Должен возвращать историю обучения (если применимо).
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Возвращает бинарные предсказания (0 или 1)."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Возвращает вероятности принадлежности к положительному классу (аномалия)."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Сохранение модели в файл."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Загрузка модели из файла."""
        pass
        