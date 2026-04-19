"""
Автоэнкодер с классификационной головой для обнаружения аномалий.
Обучается одновременно минимизировать ошибку реконструкции и кросс-энтропию.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from .base_model import BaseModel


class AutoencoderModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(name="autoencoder", **kwargs)
        default_params = {
            "encoding_dim": 32,
            "hidden_layers": [128, 64],
            "dropout_rate": 0.2,
            "activation": "relu",
            "output_activation": "sigmoid",
            "reconstruction_weight": 0.5,  # Вес ошибки реконструкции в общей функции потерь
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 256,
            "epochs": 50,
            "early_stopping_patience": 10,
        }
        default_params.update(kwargs)
        self.params = default_params
        self.history: Optional[keras.callbacks.History] = None
        self.encoder: Optional[Model] = None
        self.decoder: Optional[Model] = None
        self.classifier: Optional[Model] = None
        self.full_model: Optional[Model] = None

    def build(self, input_shape: int, **kwargs) -> None:
        self.params["input_shape"] = input_shape

        # Входной слой
        input_layer = layers.Input(shape=(input_shape,), name="input")
        x = input_layer

        # Энкодер
        for units in self.params["hidden_layers"]:
            x = layers.Dense(units, activation=self.params["activation"])(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.params["dropout_rate"])(x)
        bottleneck = layers.Dense(self.params["encoding_dim"], activation=self.params["activation"], name="bottleneck")(x)
        self.encoder = Model(inputs=input_layer, outputs=bottleneck, name="encoder")

        # Декодер
        decoder_input = layers.Input(shape=(self.params["encoding_dim"],))
        x = decoder_input
        for units in reversed(self.params["hidden_layers"]):
            x = layers.Dense(units, activation=self.params["activation"])(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.params["dropout_rate"])(x)
        reconstructed = layers.Dense(input_shape, activation="linear", name="reconstruction")(x)
        self.decoder = Model(inputs=decoder_input, outputs=reconstructed, name="decoder")

        # Классификатор (поверх bottleneck)
        x = layers.Dense(16, activation="relu")(bottleneck)
        x = layers.Dropout(0.2)(x)
        classification_output = layers.Dense(1, activation="sigmoid", name="classification")(x)

        # Полная модель с двумя выходами
        self.full_model = Model(
            inputs=input_layer,
            outputs=[reconstructed, classification_output],
            name="AE_Classifier"
        )

        # Компиляция
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
                "classification": ["accuracy", keras.metrics.AUC(name="auc")]
            }
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            **kwargs) -> keras.callbacks.History:
        if self.full_model is None:
            self.build(X_train.shape[1])

        # Подготовка целевых значений: реконструкция = вход, классификация = метка
        y_train_dict = {
            "reconstruction": X_train,
            "classification": y_train
        }
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, {"reconstruction": X_val, "classification": y_val})

        # Колбэки
        cb_list = []
        if validation_data is not None:
            early_stop = keras.callbacks.EarlyStopping(
                monitor="val_classification_loss",
                patience=self.params["early_stopping_patience"],
                restore_best_weights=True
            )
            cb_list.append(early_stop)

        log_dir = Path("reports/training_logs") / self.name
        log_dir.mkdir(parents=True, exist_ok=True)
        cb_list.append(keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1))

        self.history = self.full_model.fit(
            X_train, y_train_dict,
            batch_size=self.params["batch_size"],
            epochs=self.params["epochs"],
            validation_data=validation_data,
            callbacks=cb_list,
            verbose=kwargs.get("verbose", 1)
        )
        self.is_fitted = True
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        _, classification_output = self.full_model.predict(X, verbose=0)
        return classification_output.flatten()

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Вычисляет ошибку реконструкции (MSE) для каждого образца."""
        self._check_fitted()
        reconstructed, _ = self.full_model.predict(X, verbose=0)
        return np.mean(np.square(X - reconstructed), axis=1)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.full_model.save(path / "full_model.h5")
        with open(path / "params.json", "w") as f:
            json.dump(self.params, f, indent=2)
        if self.history is not None:
            hist_df = pd.DataFrame(self.history.history)
            hist_df.to_csv(path / "history.csv", index=False)

    @classmethod
    def load(cls, path: Path) -> "AutoencoderModel":
        with open(path / "params.json", "r") as f:
            params = json.load(f)
        instance = cls(**params)
        instance.full_model = keras.models.load_model(path / "full_model.h5")
        # Восстанавливаем энкодер и декодер как части модели
        instance.encoder = Model(
            inputs=instance.full_model.input,
            outputs=instance.full_model.get_layer("bottleneck").output
        )
        instance.is_fitted = True
        return instance

    def _check_fitted(self):
        if not self.is_fitted or self.full_model is None:
            raise RuntimeError("Model must be fitted before prediction.")
            