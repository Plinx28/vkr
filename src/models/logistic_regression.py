"""
Модель логистической регрессии (обёртка над sklearn.LogisticRegression).

Реализует интерфейс :class:`BaseModel` для линейного классификатора с
L2-регуляризацией. Используется как baseline-модель в проекте.
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from sklearn.linear_model import LogisticRegression

from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """Модель логистической регрессии.

    Обёртка над :class:`sklearn.linear_model.LogisticRegression`,
    приводящая её к единому интерфейсу проекта. Порог бинаризации по
    умолчанию вынесен в ``0.59`` и может быть уточнён через
    :meth:`~base_model.BaseModel.optimize_threshold`.

    Attributes:
        params (dict): Гиперпараметры sklearn-классификатора.
        threshold_ (float): Порог бинаризации вероятностей (по умолчанию ``0.59``).
    """

    def __init__(self, **kwargs):
        """Инициализирует модель и её гиперпараметры по умолчанию.

        Значения по умолчанию могут быть переопределены через ``kwargs``.

        Args:
            **kwargs: Гиперпараметры, переопределяющие значения по умолчанию
                (``penalty``, ``C``, ``solver``, ``max_iter``, ``tol``,
                ``class_weight``, ``random_state`` и т. п.).
        """
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
        """Инициализация sklearn-классификатора.

        Args:
            input_shape: Число входных признаков (сохраняется для справки).
            **kwargs: Не используется; добавлен для совместимости с интерфейсом.
        """
        self._input_shape = input_shape
        sk_params = self.params.copy()
        sk_params.pop('input_shape', None)
        self.model = LogisticRegression(**sk_params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            **kwargs) -> None:
        """Обучение модели.

        При необходимости создаёт внутренний классификатор и обучает его на
        тренировочной выборке. Валидационная выборка sklearn-моделью не
        используется и принимается лишь для совместимости интерфейса.

        Args:
            X_train: Матрица признаков обучающей выборки.
            y_train: Вектор меток обучающей выборки.
            X_val: Не используется (совместимость интерфейса).
            y_val: Не используется (совместимость интерфейса).
            **kwargs: Дополнительные параметры (не используются).

        Returns:
            None.
        """
        if self.model is None:
            self.build(X_train.shape[1])

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        if hasattr(self.model, 'n_iter_'):
            print(f"[LR] Converged in {self.model.n_iter_[0]} iterations.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Возвращает бинарные метки, полученные пороговой обработкой вероятностей.

        Args:
            X: Матрица признаков формы ``(n_samples, n_features)``.

        Returns:
            Массив меток ``{0, 1}`` формы ``(n_samples,)``.
        """
        self._check_fitted()
        proba = self.predict_proba(X)
        return (proba >= self.threshold_).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Возвращает вероятности положительного класса.

        Args:
            X: Матрица признаков формы ``(n_samples, n_features)``.

        Returns:
            Массив вероятностей класса ``1`` формы ``(n_samples,)``.
        """
        self._check_fitted()
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        """Сохраняет модель, её параметры и размерность входа в один файл.

        Артефакты сериализуются через ``joblib`` в файл ``model.joblib``.

        Args:
            path: Директория для сохранения (создаётся при отсутствии).
        """
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
        """Загружает модель из файла ``model.joblib``.

        Args:
            path: Директория, содержащая сохранённый файл модели.

        Returns:
            Восстановленный экземпляр :class:`LogisticRegressionModel`.
        """
        model_path = path / "model.joblib"
        data = joblib.load(model_path)
        instance = cls(**data["params"])
        instance.model = data["model"]
        instance._input_shape = data.get("input_shape")
        instance.is_fitted = True
        return instance
