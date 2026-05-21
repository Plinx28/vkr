"""
Автоэнкодер с классификационной головой и бинарной кросс‑энтропией.
Исправлен EarlyStopping (mode='min'), метка y приводится к (N,1).
"""

import json
import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers, Model
import pandas as pd

from .base_model import BaseModel


class AutoencoderModel(BaseModel):
    """Модель автоэнкодера."""

    def __init__(self, **kwargs):
        super().__init__(name="autoencoder", **kwargs)
        default_params = {
            "encoding_dim": 32,
            "hidden_layers": [64, 32],
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
                "classification": ["accuracy"]
            }
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
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
        self._check_fitted()
        proba = self.predict_proba(X)
        return (proba >= self.threshold_).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        _, class_out = self.full_model.predict(X, verbose=0)
        return class_out.flatten()

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.full_model.save(path / "full_model.h5")
        with open(path / "params.json", "w") as f:
            json.dump(self.params, f, indent=2)
        if self.history is not None:
            pd.DataFrame(self.history.history).to_csv(path / "history.csv", index=False)

    @classmethod
    def load(cls, path: Path) -> "AutoencoderModel":
        with open(path / "params.json", "r") as f:
            params = json.load(f)
        instance = cls(**params)
        instance.full_model = keras.models.load_model(path / "full_model.h5", compile=False)
        instance.encoder = Model(
            inputs=instance.full_model.input,
            outputs=instance.full_model.get_layer("bottleneck").output
        )
        instance.is_fitted = True
        return instance

    def _check_fitted(self):
        if not self.is_fitted or self.full_model is None:
            raise RuntimeError("Model must be fitted before prediction.")
