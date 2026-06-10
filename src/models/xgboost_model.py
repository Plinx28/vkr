"""
Реализация градиентного бустинга XGBoost.

Модуль предоставляет обёртку над :class:`xgboost.XGBClassifier`, приводящую её
к единому интерфейсу :class:`BaseModel`. Поддерживает автоматический учёт
дисбаланса классов через параметр ``scale_pos_weight`` и раннюю остановку по
валидационной выборке.
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import xgboost as xgb

from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """Модель градиентного бустинга над деревьями решений.

    Attributes:
        params (dict): Гиперпараметры XGBoost-классификатора.
        threshold_ (float): Порог бинаризации (наследуется от
            :class:`BaseModel`; для данной модели предсказание выполняется
            штатным методом XGBoost).
    """
    def __init__(self, **kwargs):
        """Инициализирует модель и гиперпараметры бустинга по умолчанию.

        Args:
            **kwargs: Гиперпараметры, переопределяющие значения по умолчанию
                (``n_estimators``, ``max_depth``, ``learning_rate``,
                ``subsample``, ``colsample_bytree``, ``scale_pos_weight``,
                ``early_stopping_rounds`` и т. п.).
        """
        super().__init__(name="xgboost", **kwargs)
        default_params = {
            "n_estimators": 200,
            "max_depth": 7,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": None,  # Будет вычислен автоматически при fit
            "early_stopping_rounds": 5,
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": ["logloss", "logloss"],
            "verbosity": 1,
        }
        default_params.update(kwargs)
        self.params = default_params

    def build(self, input_shape: int, **kwargs) -> None:
        """Создаёт экземпляр ``xgb.XGBClassifier`` с заданными параметрами.

        Args:
            input_shape: Число входных признаков (сохраняется для справки).
            **kwargs: Не используется; добавлен для совместимости с интерфейсом.
        """
        self._input_shape = input_shape
        sk_params = self.params.copy()
        sk_params.pop('input_shape', None)
        self.model = xgb.XGBClassifier(**sk_params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            **kwargs) -> None:
        """Обучает модель с автоматической компенсацией дисбаланса классов.

        Если ``scale_pos_weight`` не задан явно, он вычисляется как отношение
        числа отрицательных примеров к числу положительных. При наличии
        валидационной выборки она добавляется в ``eval_set`` для ранней
        остановки и мониторинга метрики.

        Args:
            X_train: Матрица признаков обучающей выборки.
            y_train: Вектор меток обучающей выборки.
            X_val: Необязательная матрица признаков валидационной выборки.
            y_val: Необязательный вектор меток валидационной выборки.
            **kwargs: Дополнительные параметры (например, ``verbose``).

        Returns:
            None.
        """
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
        """Возвращает бинарные метки, полученные пороговой обработкой вероятностей.

        Метки определяются сравнением вероятности положительного класса с
        порогом ``threshold_`` (подобранным на валидации), что обеспечивает
        единообразие поведения со всеми остальными моделями проекта.

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
        """Сохраняет модель и её параметры в директорию.

        Сама модель сохраняется в нативном формате XGBoost (``model.json``),
        а гиперпараметры — отдельно через ``joblib`` (``params.joblib``).

        Args:
            path: Директория для сохранения (создаётся при отсутствии).
        """
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / "model.json"
        self.model.save_model(model_path)
        # Сохраняем параметры отдельно
        params_path = path / "params.joblib"
        joblib.dump(self.params, params_path)
        self._save_threshold(path)

    @classmethod
    def load(cls, path: Path) -> "XGBoostModel":
        """Загружает модель из директории.

        Восстанавливает гиперпараметры из ``params.joblib`` и веса модели из
        ``model.json``.

        Args:
            path: Директория с сохранёнными артефактами модели.

        Returns:
            Восстановленный экземпляр :class:`XGBoostModel`.
        """
        model_path = path / "model.json"
        params_path = path / "params.joblib"
        params = joblib.load(params_path)
        instance = cls(**params)
        instance.model = xgb.XGBClassifier()
        instance.model.load_model(model_path)
        instance.is_fitted = True
        instance._load_threshold(path)
        return instance

    def _check_fitted(self):
        """Проверяет, что модель обучена, перед выполнением предсказания.

        Raises:
            RuntimeError: Если модель ещё не обучена.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")
