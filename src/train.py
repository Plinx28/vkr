#!/usr/bin/env python3
"""
Скрипт обучения выбранной модели с аргументами командной строки.
"""

import argparse
import logging
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from utils import set_seed, timer, get_dataset_splits, compute_metrics, print_metrics
from models.logistic_regression import LogisticRegressionModel
from models.xgboost_model import XGBoostModel
from models.mlp_model import MLPModel
from models.autoencoder_model import AutoencoderModel

# Настройка логирования
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
    return parser.parse_args()


def plot_training_history(history, model_name: str, save_dir: Path):
    """Сохраняет графики обучения для нейросетевых моделей."""
    if history is None:
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    hist_df = pd.DataFrame(history.history)

    # График потерь
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist_df["loss"], label="train_loss")
    if "val_loss" in hist_df:
        plt.plot(hist_df["val_loss"], label="val_loss")
    plt.title(f"{model_name} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # График метрик (accuracy/auc)
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

    # 1. Загрузка данных
    logger.info("Loading data...")
    with timer("Data loading"):
        X_train, X_test, y_train, y_test = get_dataset_splits(force_reload=args.no_cache)

    # 2. Создание модели
    model_class = MODEL_REGISTRY[args.model]
    params = {}
    if args.params:
        with open(args.params, "r") as f:
            params = json.load(f)
    model = model_class(**params)
    logger.info(f"Initialized {model.name} model.")

    # 3. Обучение
    output_path = Path(args.output_dir) / model.name
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training...")
    with timer("Training"):
        # Для нейросетей используем валидационную выборку (часть train)
        if isinstance(model, (MLPModel, AutoencoderModel)):
            # Выделяем 10% из тренировочных данных под валидацию
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.1, random_state=args.seed, stratify=y_train
            )
            history = model.fit(X_tr, y_tr, X_val, y_val, verbose=1)
        else:
            history = model.fit(X_train, y_train, verbose=1)

    # 4. Сохранение модели
    model.save(output_path)
    logger.info(f"Model saved to {output_path}")

    # 5. Оценка на тестовой выборке
    logger.info("Evaluating on test set...")
    with timer("Evaluation"):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba)

    print_metrics(metrics, title=f"Test Metrics for {model.name}")

    # Сохраняем метрики
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_path / "test_metrics.csv", index=False)

    # 6. Графики обучения (если есть история)
    if history is not None:
        plot_training_history(history, model.name, Path("reports/figures"))

    logger.info("Done.")


if __name__ == "__main__":
    main()
    