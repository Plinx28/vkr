"""
Автоэнкодер с классификационной головой и бинарной кросс‑энтропией.

Модуль реализует гибридную нейросетевую модель на базе Keras: автоэнкодер
(энкодер + декодер) с дополнительной классификационной головой,
подключённой к бутылочному горлышку (bottleneck). Модель обучается на двух
задачах одновременно — реконструкции входа (MSE) и бинарной классификации
(binary crossentropy), что задаётся взвешенной суммой потерь.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, Model

from .base_model import BaseModel


class AutoencoderModel(BaseModel):
    """Модель автоэнкодера.

    Сеть состоит из энкодера, сжимающего вход в латентное представление,
    декодера, восстанавливающего вход, и классификационной головы,
    предсказывающей метку класса по латентному представлению. Обучение
    ведётся одновременно по двум выходам с весами потерь, заданными
    параметром ``reconstruction_weight``.

    Attributes:
        params (dict): Гиперпараметры модели (архитектура, обучение и т. п.).
        history: История обучения, заполняемая после :meth:`fit`.
        encoder (keras.Model): Подсеть-энкодер (вход → bottleneck).
        full_model (keras.Model): Полная модель с двумя выходами
            (реконструкция и классификация).
    """

    def __init__(self, **kwargs):
        """Инициализирует модель и гиперпараметры по умолчанию.

        Args:
            **kwargs: Гиперпараметры, переопределяющие значения по умолчанию
                (``encoding_dim``, ``hidden_layers``, ``dropout_rate``,
                ``activation``, ``reconstruction_weight``, ``optimizer``,
                ``learning_rate``, ``batch_size``, ``epochs``,
                ``early_stopping_patience`` и др.).
        """
        super().__init__(name="autoencoder", **kwargs)
        default_params = {
            "encoding_dim": 32,
            "hidden_layers": [32, 16],
            "dropout_rate": 0.2,
            "activation": "relu",
            "output_activation": "sigmoid",
            "reconstruction_weight": 0.5,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 256,
            "epochs": 13,
            "early_stopping_patience": 2,
            "focal_gamma": 2.0,        # не используется
        }
        default_params.update(kwargs)
        self.params = default_params
        self.history = None
        self.encoder = None
        self.full_model = None

    def build(self, input_shape: int, **kwargs) -> None:
        """Строит и компилирует архитектуру автоэнкодера с классификатором.

        Формирует три части сети: энкодер (со слоями Dense, BatchNorm,
        Dropout, сжимающими вход до ``encoding_dim``), симметричный декодер с
        линейным выходом реконструкции и классификационную голову. Модель
        компилируется с двумя функциями потерь (MSE для реконструкции и
        binary crossentropy для классификации), взвешенными согласно
        ``reconstruction_weight``.

        Args:
            input_shape: Число входных признаков (оно же — размер реконструкции).
            **kwargs: Не используется; добавлен для совместимости с интерфейсом.
        """
        self.params["input_shape"] = input_shape

        input_layer = layers.Input(shape=(input_shape,), name="input")
        x = input_layer

        # Энкодер
        for units in self.params["hidden_layers"]:
            x = layers.Dense(units, activation=self.params["activation"])(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.params["dropout_rate"])(x)
        bottleneck = layers.Dense(self.params["encoding_dim"],
                                  activation=self.params["activation"],
                                  name="bottleneck")(x)

        self.encoder = Model(inputs=input_layer, outputs=bottleneck, name="encoder")

        # Декодер
        x = bottleneck
        for units in reversed(self.params["hidden_layers"]):
            x = layers.Dense(units, activation=self.params["activation"])(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.params["dropout_rate"])(x)
        reconstructed = layers.Dense(input_shape, activation="linear",
                                     name="reconstruction")(x)

        # Классификационная голова
        x_cls = layers.Dense(16, activation="relu")(bottleneck)
        x_cls = layers.Dropout(0.2)(x_cls)
        classification_output = layers.Dense(1, activation="sigmoid",
                                             name="classification")(x_cls)

        self.full_model = Model(
            inputs=input_layer,
            outputs=[reconstructed, classification_output],
            name="AE_Classifier"
        )

        optimizer = keras.optimizers.get({
            "class_name": self.params["optimizer"],
            "config": {"learning_rate": self.params["learning_rate"]}
        })

        self.full_model.compile(
            optimizer=optimizer,
            loss={
                "reconstruction": "mse",
                "classification": "binary_crossentropy"
            },
            loss_weights={
                "reconstruction": self.params["reconstruction_weight"],
                "classification": 1.0 - self.params["reconstruction_weight"]
            },
            metrics={
                "classification": [
                    keras.metrics.AUC(name="auc"),
                    keras.metrics.Precision(name="precision"),
                    keras.metrics.Recall(name="recall")
                ]
            }
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Обучает модель на двух задачах: реконструкции и классификации.

        Целевыми значениями выступают сам вход ``X_train`` (для выхода
        реконструкции) и метки ``y_train`` (для классификационного выхода).
        При наличии валидационной выборки подключается ранняя остановка по
        ``val_classification_loss``; также ведётся логирование в TensorBoard.

        Args:
            X_train: Матрица признаков обучающей выборки.
            y_train: Вектор меток обучающей выборки.
            X_val: Необязательная матрица признаков валидационной выборки.
            y_val: Необязательный вектор меток валидационной выборки.
            **kwargs: Дополнительные параметры (например, ``verbose``).

        Returns:
            keras.callbacks.History: История обучения модели.
        """
        if self.full_model is None:
            self.build(X_train.shape[1])

        y_train_cls = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
        y_train_dict = {
            "reconstruction": X_train,
            "classification": y_train_cls
        }

        validation_data = None
        callbacks_list = []

        if X_val is not None and y_val is not None:
            y_val_cls = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
            validation_data = (X_val, {"reconstruction": X_val, "classification": y_val_cls})

            early_stop = keras.callbacks.EarlyStopping(
                monitor="val_classification_loss",
                patience=self.params["early_stopping_patience"],
                restore_best_weights=True,
                mode="min"                     # Явно указываем минимизацию
            )
            callbacks_list.append(early_stop)

        log_dir = Path("reports/training_logs") / self.name
        log_dir.mkdir(parents=True, exist_ok=True)
        callbacks_list.append(keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1))

        self.history = self.full_model.fit(
            X_train, y_train_dict,
            batch_size=self.params["batch_size"],
            epochs=self.params["epochs"],
            validation_data=validation_data,
            callbacks=callbacks_list,
            verbose=kwargs.get("verbose", 1)
        )
        self.is_fitted = True
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Возвращает бинарные метки на основе классификационного выхода.

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

        Из двух выходов модели используется только классификационный; выход
        реконструкции игнорируется.

        Args:
            X: Матрица признаков формы ``(n_samples, n_features)``.

        Returns:
            Одномерный массив вероятностей формы ``(n_samples,)``.
        """
        self._check_fitted()
        _, class_out = self.full_model.predict(X, verbose=0)
        return class_out.flatten()

    def save(self, path: Path) -> None:
        """Сохраняет полную модель, гиперпараметры и историю обучения.

        Полная модель сохраняется в ``full_model.h5``, гиперпараметры — в
        ``params.json``, история обучения (при наличии) — в ``history.csv``.

        Args:
            path: Директория для сохранения (создаётся при отсутствии).
        """
        path.mkdir(parents=True, exist_ok=True)
        self.full_model.save(path / "full_model.h5")
        with open(path / "params.json", "w") as f:
            json.dump(self.params, f, indent=2)
        if self.history is not None:
            pd.DataFrame(self.history.history).to_csv(path / "history.csv", index=False)
        self._save_threshold(path)

    @classmethod
    def load(cls, path: Path) -> "AutoencoderModel":
        """Загружает модель из директории.

        Восстанавливает гиперпараметры из ``params.json`` и полную модель из
        ``full_model.h5``, после чего отдельно собирает подсеть-энкодер на
        основе слоя ``bottleneck``.

        Args:
            path: Директория с сохранёнными артефактами модели.

        Returns:
            Восстановленный экземпляр :class:`AutoencoderModel`.
        """
        with open(path / "params.json", "r") as f:
            params = json.load(f)
        instance = cls(**params)
        instance.full_model = keras.models.load_model(path / "full_model.h5", compile=False)
        instance.encoder = Model(
            inputs=instance.full_model.input,
            outputs=instance.full_model.get_layer("bottleneck").output
        )
        instance.is_fitted = True
        instance._load_threshold(path)
        return instance

    def _check_fitted(self):
        """Проверяет, что модель обучена, перед выполнением предсказания.

        Raises:
            RuntimeError: Если модель ещё не обучена.
        """
        if not self.is_fitted or self.full_model is None:
            raise RuntimeError("Model must be fitted before prediction.")
