"""
Многослойный перцептрон (MLP) с фокальной потерей и подбором порога.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import pandas as pd

from .base_model import BaseModel


class MLPModel(BaseModel):
    """Модель многослойного перцептрона."""
    def __init__(self, **kwargs):
        super().__init__(name="mlp", **kwargs)
        default_params: Dict[str, Any] = {
            "hidden_layers": [128, 64, 32],
            "dropout_rate": 0.3,
            "activation": "relu",
            "output_activation": "sigmoid",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 256,
            "epochs": 15,
            "early_stopping_patience": 3,
            "focal_gamma": 2.0,
            "class_weight": None,
        }
        default_params.update(kwargs)
        self.params = default_params
        self.history: Optional[keras.callbacks.History] = None

    def build(self, input_shape: int, **kwargs) -> None:
        self.params["input_shape"] = input_shape
        model = keras.Sequential(name="MLP_Classifier")
        model.add(layers.Input(shape=(input_shape,)))

        for units in self.params["hidden_layers"]:
            model.add(layers.Dense(units, activation=self.params["activation"]))
            model.add(layers.Dropout(self.params["dropout_rate"]))
            model.add(layers.BatchNormalization())

        model.add(layers.Dense(1, activation=self.params["output_activation"]))

        optimizer = keras.optimizers.get({
            "class_name": self.params["optimizer"],
            "config": {"learning_rate": self.params["learning_rate"]}
        })
        loss = tf.keras.losses.BinaryFocalCrossentropy(
            gamma=self.params.get("focal_gamma", 2.0)
        )
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["accuracy", keras.metrics.AUC(name="auc"),
                     keras.metrics.Precision(name="precision"),
                     keras.metrics.Recall(name="recall")]
        )
        self.model = model

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        if self.model is None:
            self.build(X_train.shape[1])

        class_weight = self.params.get("class_weight")
        if class_weight is None:
            neg, pos = np.bincount(y_train)
            total = neg + pos
            weight_for_0 = (1 / neg) * (total / 2.0)
            weight_for_1 = (1 / pos) * (total / 2.0)
            class_weight = {0: weight_for_0, 1: weight_for_1}

        cb_list = []
        if X_val is not None and y_val is not None:
            early_stop = callbacks.EarlyStopping(
                monitor="val_loss", patience=self.params["early_stopping_patience"],
                restore_best_weights=True
            )
            cb_list.append(early_stop)

        checkpoint_path = Path("models") / self.name / "checkpoint.h5"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        cb_list.append(callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss" if X_val is not None else "loss",
            save_best_only=True
        ))

        log_dir = Path("reports/training_logs") / self.name
        log_dir.mkdir(parents=True, exist_ok=True)
        cb_list.append(callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1))

        validation_data = (X_val, y_val) if X_val is not None else None

        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.params["batch_size"],
            epochs=self.params["epochs"],
            validation_data=validation_data,
            class_weight=class_weight,
            callbacks=cb_list,
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
        return self.model.predict(X, verbose=0).flatten()

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "model.h5")
        with open(path / "params.json", "w") as f:
            json.dump(self.params, f, indent=2)
        if self.history is not None:
            pd.DataFrame(self.history.history).to_csv(path / "history.csv", index=False)

    @classmethod
    def load(cls, path: Path) -> "MLPModel":
        with open(path / "params.json", "r") as f:
            params = json.load(f)
        instance = cls(**params)
        instance.model = keras.models.load_model(path / "model.h5")
        instance.is_fitted = True
        return instance

    def _check_fitted(self):
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")
