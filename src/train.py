"""
Скрипт обучения моделей на полных данных из data/train, с валидацией на data/val.

Запускается из командной строки. Выбирает модель по имени из реестра
``MODEL_REGISTRY``, при необходимости подгружает гиперпараметры из JSON-файла,
загружает обучающую и валидационную выборки, обучает модель, подбирает порог
бинаризации и сохраняет артефакты. Для нейросетевых моделей дополнительно
строит графики истории обучения.

Пример запуска:
    python train.py --model xgboost --params configs/xgb.json --seed 42
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from models.autoencoder_model import AutoencoderModel
from models.logistic_regression import LogisticRegressionModel
from models.mlp_model import MLPModel
from models.xgboost_model import XGBoostModel
from utils import set_seed, timer, load_data_from_dir

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Реестр доступных моделей: сопоставляет имя из CLI с классом модели.
# Для каждой модели предусмотрены краткий и полный псевдонимы.
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
    """Разбирает аргументы командной строки.

    Returns:
        argparse.Namespace: Объект с полями ``model`` (имя модели из реестра),
        ``params`` (путь к JSON с гиперпараметрами или ``None``),
        ``output_dir`` (директория для сохранения) и ``seed`` (значение seed).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, choices=MODEL_REGISTRY.keys()
    )
    parser.add_argument("--params", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def plot_training_history(history, model_name, save_dir) -> None:
    """Строит и сохраняет графики истории обучения нейросетевой модели.

    Формирует два подграфика: динамику функции потерь (train/val) и динамику
    остальных метрик по эпохам. Результат сохраняется в PNG-файл. Если
    история не передана, функция ничего не делает.

    Args:
        history: Объект истории обучения Keras (с атрибутом ``.history``)
            либо ``None``.
        model_name: Имя модели, используемое в заголовках и имени файла.
        save_dir (Path): Директория для сохранения изображения.
    """
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
    """Точка входа: полный цикл обучения одной модели.

    Последовательно выполняет: фиксацию seed, создание модели с
    гиперпараметрами, загрузку обучающей и валидационной выборок, обучение,
    подбор порога на валидации, сохранение модели.
    """
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
        model.optimize_threshold(X_val, y_val)

    model.save(output_path)

    if isinstance(model, (MLPModel, AutoencoderModel)) and history is not None:
        plot_training_history(history, model.name, Path("reports/figures"))

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
