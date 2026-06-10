"""
Абстрактный базовый класс с унифицированным интерфейсом и подбором порога.

Модуль определяет общий контракт для всех моделей проекта (логистическая
регрессия, XGBoost, MLP, автоэнкодер). Любая конкретная модель наследует
:class:`BaseModel` и реализует абстрактные методы построения, обучения,
предсказания, сохранения и загрузки. Это позволяет скриптам обучения и оценки
(``train.py``, ``evaluate.py``) работать с моделями единообразно, не завися
от деталей конкретной реализации.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef


class BaseModel(ABC):
    """Абстрактный базовый класс для моделей.

    Задаёт единый интерфейс, которому следуют все модели в проекте, и
    реализует общий для них механизм подбора порога бинаризации
    (:meth:`optimize_threshold`).

    Attributes:
        name (str): Короткое имя модели; используется как имя папки при
            сохранении и как ключ в реестрах моделей.
        model (Any): Внутренний объект конкретной модели (например, sklearn-
            или Keras-модель). Инициализируется в :meth:`build`/:meth:`fit`.
        params (dict): Словарь гиперпараметров модели.
        is_fitted (bool): Флаг того, что модель обучена и готова к предсказанию.
        threshold_ (float): Порог бинаризации вероятностей в метки классов.
            По умолчанию ``0.5``; может быть переопределён в
            :meth:`optimize_threshold`.
    """
    def __init__(self, name: str, **kwargs):
        """Инициализирует базовые атрибуты модели.

        Args:
            name: Короткое имя модели (используется при сохранении/загрузке).
            **kwargs: Произвольные гиперпараметры, сохраняемые в ``self.params``.
        """
        self.name = name
        self.model: Any = None
        self.params = kwargs
        self.is_fitted = False
        self.threshold_ = 0.5

    @abstractmethod
    def build(self, input_shape: int, **kwargs) -> None:
        """Создаёт и инициализирует внутреннюю модель.

        Args:
            input_shape: Число входных признаков (размерность вектора объекта).
            **kwargs: Дополнительные параметры построения, специфичные для модели.
        """
        pass

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            **kwargs) -> Any:
        """Обучает модель на тренировочной выборке.

        Args:
            X_train: Матрица признаков обучающей выборки формы ``(n_samples, n_features)``.
            y_train: Вектор бинарных меток обучающей выборки формы ``(n_samples,)``.
            X_val: Необязательная матрица признаков валидационной выборки.
            y_val: Необязательный вектор меток валидационной выборки.
            **kwargs: Дополнительные параметры обучения (например, ``verbose``).

        Returns:
            Объект истории обучения (для нейросетевых моделей) либо ``None``.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Возвращает предсказанные бинарные метки классов.

        Args:
            X: Матрица признаков формы ``(n_samples, n_features)``.

        Returns:
            Массив меток ``{0, 1}`` формы ``(n_samples,)``.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Возвращает оценки вероятности положительного класса.

        Args:
            X: Матрица признаков формы ``(n_samples, n_features)``.

        Returns:
            Массив вероятностей в диапазоне ``[0, 1]`` формы ``(n_samples,)``.
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Сохраняет модель и её параметры в указанную директорию.

        Args:
            path: Путь к директории для сохранения артефактов модели.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Загружает ранее сохранённую модель из директории.

        Args:
            path: Путь к директории с артефактами модели.

        Returns:
            Восстановленный экземпляр модели, готовый к предсказанию.
        """
        pass

    def optimize_threshold(self, X_val, y_val, metric: str = "f1", beta: float = 2.0) -> float:
        probas = self.predict_proba(X_val)
        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = []
        for thr in thresholds:
            preds = (probas >= thr).astype(int)
            if metric == "f1":
                scores.append(f1_score(y_val, preds, zero_division=0))
            elif metric == "fbeta":
                scores.append(fbeta_score(y_val, preds, beta=beta, zero_division=0))
            elif metric == "mcc":
                scores.append(matthews_corrcoef(y_val, preds))
        best_idx = int(np.argmax(scores))
        self.threshold_ = float(thresholds[best_idx])
        print(f"optimized threshold {self.threshold_} ({metric})")
        return self.threshold_

    def _save_threshold(self, path: Path) -> None:
        """Сохраняет подобранный порог бинаризации в файл ``threshold.json``.

        Метод вызывается из :meth:`save` конкретных моделей, чтобы порог,
        найденный на валидации в :meth:`optimize_threshold`, сохранялся вместе
        с моделью и не сбрасывался к значению по умолчанию при загрузке.

        Args:
            path: Директория, в которую сохраняется модель.
        """
        with open(Path(path) / "threshold.json", "w") as f:
            json.dump({"threshold": float(self.threshold_)}, f)

    def _load_threshold(self, path: Path) -> None:
        """Восстанавливает порог бинаризации из файла ``threshold.json``.

        Если файл отсутствует (например, модель сохранена старой версией кода),
        порог не изменяется и сохраняет значение по умолчанию, заданное в
        конструкторе модели.

        Args:
            path: Директория, из которой загружается модель.
        """
        threshold_path = Path(path) / "threshold.json"
        if threshold_path.exists():
            with open(threshold_path, "r") as f:
                self.threshold_ = json.load(f)["threshold"]
