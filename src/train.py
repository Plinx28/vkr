#!/usr/bin/env python3
"""
Скрипт обучения с потоковой загрузкой для нейросетей и с подвыборкой для классических моделей.
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
from utils import (
    set_seed, timer, get_dataset_splits, compute_metrics, print_metrics,
    CSVDataGenerator, MetricsCallback, find_optimal_threshold
)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=MODEL_REGISTRY.keys())
    parser.add_argument("--params", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples for classical models (ignored for NN).")
    parser.add_argument("--use_generator", action="store_true",
                        help="Stream data from CSV files (required for full dataset).")
    return parser.parse_args()

def plot_training_history(history, model_name, save_dir):
    if history is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    hist_df = pd.DataFrame(history.history)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(hist_df["loss"], label="train_loss")
    if "val_loss" in hist_df:
        plt.plot(hist_df["val_loss"], label="val_loss")
    plt.title(f"{model_name} - Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    metric_cols = [c for c in hist_df.columns if c not in ("loss", "val_loss")]
    for col in metric_cols:
        plt.plot(hist_df[col], label=col)
    plt.title(f"{model_name} - Metrics")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_history.png", dpi=150)
    plt.close()
    logger.info(f"Training plots saved to {save_dir}")

def main():
    args = parse_args()
    set_seed(args.seed)

    model_class = MODEL_REGISTRY[args.model]
    params = {}
    if args.params:
        with open(args.params, "r") as f:
            params = json.load(f)
    model = model_class(**params)

    output_path = Path(args.output_dir) / model.name
    output_path.mkdir(parents=True, exist_ok=True)

    if args.use_generator and isinstance(model, (MLPModel, AutoencoderModel)):
        # Хронологическое разделение файлов
        csv_files = sorted(Path("data/processed").glob("*.csv"))
        split_idx = int(len(csv_files) * 0.8)
        train_files = csv_files[:split_idx]
        val_files = csv_files[split_idx:]
        batch_size = model.params.get("batch_size", 256)

        train_gen = CSVDataGenerator(train_files, batch_size=batch_size, shuffle=False)
        val_gen = CSVDataGenerator(val_files, batch_size=batch_size, shuffle=False)

        sample_X, _ = train_gen[0]
        input_dim = sample_X.shape[1]

        if isinstance(model, MLPModel):
            model.build(input_dim)
            keras_model = model.model
        else:
            model.build(input_dim)
            keras_model = model.full_model

        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=model.params.get("early_stopping_patience", 10),
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(output_path / "best_model.h5"),
                monitor='val_loss', save_best_only=True
            ),
            MetricsCallback(validation_data=val_gen, threshold=0.5)
        ]

        with timer("Training"):
            history = keras_model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=model.params.get("epochs", 50),
                callbacks=callbacks_list,
                verbose=1
            )

        keras_model.load_weights(str(output_path / "best_model.h5"))
        model.is_fitted = True
        model.history = history

        # Подбор оптимального порога
        y_val_true, y_val_proba = [], []
        for Xb, yb in val_gen:
            proba = model.predict_proba(Xb)
            y_val_proba.extend(proba)
            y_val_true.extend(yb)
        opt_thresh = find_optimal_threshold(np.array(y_val_true), np.array(y_val_proba))
        model.threshold_ = opt_thresh
        logger.info(f"Optimal threshold set to {opt_thresh:.4f}")

        model.save(output_path)

        # Оценка на валидационной выборке (можно заменить на отдельный тестовый набор)
        y_test_true, y_test_pred, y_test_proba = [], [], []
        for Xb, yb in val_gen:
            proba = model.predict_proba(Xb)
            pred = (proba >= model.threshold_).astype(int)
            y_test_proba.extend(proba)
            y_test_pred.extend(pred)
            y_test_true.extend(yb)
        metrics = compute_metrics(np.array(y_test_true), np.array(y_test_pred), np.array(y_test_proba))
        print_metrics(metrics, title=f"Test Metrics for {model.name}")
        pd.DataFrame([metrics]).to_csv(output_path / "test_metrics.csv", index=False)

        plot_training_history(history, model.name, Path("reports/figures"))

    else:
        # Классический режим с подвыборкой
        if args.max_samples is None:
            logger.warning("No --max_samples specified for classical model, may run out of memory.")
        with timer("Data loading"):
            X_train, X_test, y_train, y_test = get_dataset_splits(
                force_reload=args.no_cache,
                max_samples=args.max_samples
            )
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=args.seed, stratify=y_train
        )

        if isinstance(model, (MLPModel, AutoencoderModel)):
            with timer("Training"):
                history = model.fit(X_tr, y_tr, X_val, y_val, verbose=1)
        else:
            with timer("Training"):
                history = model.fit(X_tr, y_tr, X_val, y_val, verbose=1)

        # Подбор порога
        model.tune_threshold(X_val, y_val)
        logger.info(f"Optimal threshold: {model.threshold_:.4f}")

        model.save(output_path)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba)
        print_metrics(metrics, title=f"Test Metrics for {model.name}")
        pd.DataFrame([metrics]).to_csv(output_path / "test_metrics.csv", index=False)

        if isinstance(model, (MLPModel, AutoencoderModel)) and history is not None:
            plot_training_history(history, model.name, Path("reports/figures"))

    logger.info("Done.")

if __name__ == "__main__":
    main()