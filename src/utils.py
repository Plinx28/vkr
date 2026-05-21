"""
Вспомогательные функции: фиксация seed, таймеры, метрики, загрузка данных,
callback метрик валидации, подбор порога.
"""

import os
import time
import random
import logging
from pathlib import Path
from typing import Optional, Dict
from contextlib import contextmanager

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)


@contextmanager
def timer(name: str = "Operation", log_level: int = logging.INFO):
    start = time.perf_counter()
    logger.log(log_level, f"{name} started...")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.log(log_level, f"{name} completed in {elapsed:.3f} seconds")


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.perf_counter()
        return self

    def stop(self):
        self.end_time = time.perf_counter()
        return self.elapsed

    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return end - self.start_time


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    print(f"\n{title}")
    print("-" * 40)
    for name, value in metrics.items():
        print(f"{name:12s}: {value:.4f}")


def load_data_from_dir(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Загружает все CSV-файлы из указанной папки в один массив X и y.
    """
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    logger.info(f"Loading data from {data_dir}, {len(csv_files)} files...")
    dfs = []
    for fpath in csv_files:
        df = pd.read_csv(fpath)
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    y = full["Label"].values.astype(int)
    X = full.drop(columns=["Label"]).values.astype(np.float32)
    logger.info(f"Loaded X: {X.shape}, y: {y.shape}")
    return X, y
