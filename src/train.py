#!/usr/bin/env python3
"""
Скрипт обучения моделей на полных данных из data/train, с валидацией на data/val.
Тестовые данные не используются (оставлены для evaluate.py).
"""

import argparse
import logging
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from utils import (set_seed, timer, load_data_from_dir)
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

    # Загрузка данных
    train_dir = Path("data/train")
    val_dir = Path("data/val")
    with timer("Loading train data"):
        X_train, y_train = load_data_from_dir(train_dir)
    with timer("Loading validation data"):
        X_val, y_val = load_data_from_dir(val_dir)

    # Обучение
    with timer("Training"):
        history = model.fit(X_train, y_train, X_val, y_val, verbose=1)

    model.save(output_path)

    if isinstance(model, (MLPModel, AutoencoderModel)) and history is not None:
        plot_training_history(history, model.name, Path("reports/figures"))

    logger.info("Training finished.")

if __name__ == "__main__":
    main()
