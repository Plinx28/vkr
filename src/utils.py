"""
Вспомогательные функции для проекта:
- фиксация случайных seed для воспроизводимости
- таймеры выполнения
- вычисление метрик классификации
- загрузка и разделение подготовленных данных
"""

import os
import time
import random
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix
)

# Попытка импорта TensorFlow для фиксации seed (если установлен)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Seed фиксация
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Фиксирует все источники случайности для воспроизводимости.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if TF_AVAILABLE:
        tf.random.set_seed(seed)
        # Дополнительные настройки для детерминизма GPU (если есть)
        # tf.config.experimental.enable_op_determinism()


# ─────────────────────────────────────────────────────────────────────────────
# Таймеры
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def timer(name: str = "Operation", log_level: int = logging.INFO):
    """
    Контекстный менеджер для измерения времени выполнения блока кода.
    """
    start = time.perf_counter()
    logger.log(log_level, f"{name} started...")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.log(log_level, f"{name} completed in {elapsed:.3f} seconds")


class Timer:
    """Класс-таймер для ручного запуска/остановки."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Метрики
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                    y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Вычисляет набор метрик бинарной классификации.
    Возвращает словарь с ключами: accuracy, precision, recall, f1, mcc,
    roc_auc (если переданы вероятности), pr_auc (если переданы вероятности).
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
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
    """Форматированный вывод метрик."""
    print(f"\n{title}")
    print("-" * 40)
    for name, value in metrics.items():
        print(f"{name:12s}: {value:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Загрузка и подготовка данных
# ─────────────────────────────────────────────────────────────────────────────

def load_all_processed_data(data_dir: Path = Path("data/processed")) -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает все CSV файлы из data/processed и объединяет их в единые X и y.
    Предполагается, что колонка 'Label' содержит целевую переменную (0/1).
    """
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    logger.info(f"Loading {len(csv_files)} processed CSV files...")
    dfs = []
    for fpath in csv_files:
        df = pd.read_csv(fpath)
        dfs.append(df)
        logger.debug(f"Loaded {fpath.name}: shape={df.shape}")

    full_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total samples: {full_df.shape[0]}, features: {full_df.shape[1]-1}")

    if "Label" not in full_df.columns:
        raise KeyError("Column 'Label' not found in data")

    y = full_df["Label"].values.astype(int)
    X = full_df.drop(columns=["Label"]).values.astype(np.float32)

    return X, y


def split_data(X: np.ndarray, y: np.ndarray,
               test_size: float = 0.2,
               random_state: int = 42,
               stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Разделение данных на обучающую и тестовую выборки.
    По умолчанию стратифицированное для сохранения пропорции классов.
    """
    stratify_arg = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )
    logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    logger.info(f"Train positive ratio: {y_train.mean():.4f}, Test positive ratio: {y_test.mean():.4f}")
    return X_train, X_test, y_train, y_test


def get_dataset_splits(force_reload: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Возвращает X_train, X_test, y_train, y_test.
    Можно кэшировать результаты в .npy файлы для ускорения.
    """
    cache_dir = Path("data/splits")
    cache_dir.mkdir(parents=True, exist_ok=True)
    X_train_path = cache_dir / "X_train.npy"
    X_test_path = cache_dir / "X_test.npy"
    y_train_path = cache_dir / "y_train.npy"
    y_test_path = cache_dir / "y_test.npy"

    if not force_reload and all(p.exists() for p in [X_train_path, X_test_path, y_train_path, y_test_path]):
        logger.info("Loading cached splits...")
        X_train = np.load(X_train_path)
        X_test = np.load(X_test_path)
        y_train = np.load(y_train_path)
        y_test = np.load(y_test_path)
        return X_train, X_test, y_train, y_test

    logger.info("Loading and splitting data from processed files...")
    X, y = load_all_processed_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Сохраняем для последующего использования
    np.save(X_train_path, X_train)
    np.save(X_test_path, X_test)
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)
    logger.info("Splits saved to disk.")

    return X_train, X_test, y_train, y_test