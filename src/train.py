#!/usr/bin/env python3
"""
Скрипт обучения модели с поддержкой подвыборки и генераторов.
"""

import argparse
import logging
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

from utils import (set_seed, timer, get_dataset_splits, compute_metrics, print_metrics, CSVDataGenerator)
from models.logistic_regression import LogisticRegressionModel
from models.xgboost_model import XGBoostModel
from models.mlp_model import MLPModel
from models.autoencoder_model import AutoencoderModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "lr": LogisticRegressionModel,
    "logistic_regression": LogisticRegressionModel,
    "xgb": XGBoostModel,
    "xgboost": XGBoostModel,
    "mlp": MLPModel,
    "ae": AutoencoderModel,
    "autoencoder": AutoencoderModel,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for network anomaly detection.")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model type to train.")
    parser.add_argument("--params", type=str, default=None,
                        help="Path to JSON file with model hyperparameters.")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save trained model.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--no_cache", action="store_true",
                        help="Force reload and split data, ignoring cache.")
    parser.add_argument("--max_samples", type=int, default=500_000,
                        help="Maximum number of samples to use (for classical models).")
    parser.add_argument("--use_generator", action="store_true",
                        help="Use data generator for neural networks (streaming from disk).")
    return parser.parse_args()

def plot_training_history(history, model_name: str, save_dir: Path):
    if history is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    hist_df = pd.DataFrame(history.history)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist_df["loss"], label="train_loss")
    if "val_loss" in hist_df:
        plt.plot(hist_df["val_loss"], label="val_loss")
    plt.title(f"{model_name} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    metric_cols = [c for c in hist_df.columns if "accuracy" in c or "auc" in c]
    for col in metric_cols:
        plt.plot(hist_df[col], label=col)
    plt.title(f"{model_name} - Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_history.png", dpi=150)
    plt.close()
    logger.info(f"Training plots saved to {save_dir}")

def main():
    args = parse_args()
    set_seed(args.seed)

    logger.info("Loading data...")
    model_class = MODEL_REGISTRY[args.model]
    params = {}
    if args.params:
        with open(args.params, "r") as f:
            params = json.load(f)
    model = model_class(**params)

    output_path = Path(args.output_dir) / model.name
    output_path.mkdir(parents=True, exist_ok=True)

    if args.use_generator and isinstance(model, (MLPModel, AutoencoderModel)):
        # Режим потоковой загрузки данных
        csv_files = sorted(Path("data/processed").glob("*.csv"))
        # Разделение файлов на train/validation (например, 80% / 20%)
        train_files, val_files = train_test_split(csv_files, test_size=0.2, random_state=args.seed)
        batch_size = model.params.get("batch_size", 256)

        train_gen = CSVDataGenerator(train_files, batch_size=batch_size, shuffle=True, seed=args.seed)
        val_gen = CSVDataGenerator(val_files, batch_size=batch_size, shuffle=False, seed=args.seed)

        logger.info(f"Training with generator: {len(train_files)} train files, {len(val_files)} val files")

        # Для MLP и Autoencoder нужно по-разному строить модель перед fit
        if isinstance(model, MLPModel):
            # Нужно узнать input_shape. Можно получить из первого батча
            sample_X, _ = train_gen[0]
            input_dim = sample_X.shape[1]
            model.build(input_dim)
            keras_model = model.model
        else:  # AutoencoderModel
            sample_X, _ = train_gen[0]
            input_dim = sample_X.shape[1]
            model.build(input_dim)
            keras_model = model.full_model

        # Колбэки
        callbacks_list = []
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=model.params.get("early_stopping_patience", 10),
            restore_best_weights=True
        )
        callbacks_list.append(early_stop)
        checkpoint_path = output_path / "best_model.h5"
        callbacks_list.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), monitor='val_loss', save_best_only=True
        ))

        with timer("Training"):
            history = keras_model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=model.params.get("epochs", 50),
                callbacks=callbacks_list,
                verbose=1
            )

        # Загружаем лучшую модель
        keras_model.load_weights(str(checkpoint_path))
        model.is_fitted = True
        model.history = history

        # Сохранение модели
        model.save(output_path)
        logger.info(f"Model saved to {output_path}")

        # Оценка на тестовой выборке: создадим отдельный генератор для теста (например, 10% от всех файлов)
        # Для простоты возьмём подвыборку из val файлов как тест
        test_gen = CSVDataGenerator(val_files, batch_size=batch_size, shuffle=False)
        logger.info("Evaluating on test set...")
        y_pred = []
        y_true = []
        for X_batch, y_batch in test_gen:
            pred = (keras_model.predict(X_batch, verbose=0) > 0.5).astype(int).flatten()
            y_pred.extend(pred)
            y_true.extend(y_batch)
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        y_proba = keras_model.predict(test_gen, verbose=0).flatten()
        metrics = compute_metrics(y_true, y_pred, y_proba)
        print_metrics(metrics, title=f"Test Metrics for {model.name}")
        pd.DataFrame([metrics]).to_csv(output_path / "test_metrics.csv", index=False)

        plot_training_history(history, model.name, Path("reports/figures"))

    else:
        # Классический режим с загрузкой подвыборки в память
        with timer("Data loading"):
            X_train, X_test, y_train, y_test = get_dataset_splits(
                force_reload=args.no_cache,
                max_samples=args.max_samples
            )
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        if isinstance(model, (MLPModel, AutoencoderModel)):
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.1, random_state=args.seed, stratify=y_train
            )
            with timer("Training"):
                history = model.fit(X_tr, y_tr, X_val, y_val, verbose=1)
        else:
            with timer("Training"):
                history = model.fit(X_train, y_train, verbose=1)

        model.save(output_path)
        logger.info(f"Model saved to {output_path}")

        logger.info("Evaluating on test set...")
        with timer("Evaluation"):
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            metrics = compute_metrics(y_test, y_pred, y_proba)
        print_metrics(metrics, title=f"Test Metrics for {model.name}")
        pd.DataFrame([metrics]).to_csv(output_path / "test_metrics.csv", index=False)

        if history is not None:
            plot_training_history(history, model.name, Path("reports/figures"))

    logger.info("Done.")


if __name__ == "__main__":
    main()
