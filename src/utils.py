"""
Вспомогательные функции с поддержкой работы с большими данными:
- загрузка случайной подвыборки
- генератор батчей для нейросетей
"""

import os
import time
import random
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from contextlib import contextmanager

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Seed фиксация
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)

# ─────────────────────────────────────────────────────────────────────────────
# Таймеры
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# Метрики
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                    y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
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
    print(f"\n{title}")
    print("-" * 40)
    for name, value in metrics.items():
        print(f"{name:12s}: {value:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Работа с данными
# ─────────────────────────────────────────────────────────────────────────────

def load_sample_data(data_dir: Path = Path("data/processed"),
                     max_samples: int = 1_000_000,
                     random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает случайную подвыборку из CSV-файлов, не загружая всё в память.
    Использует reservoir sampling для больших файлов.
    """
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    # Определим общее количество строк (быстрый подсчёт)
    total_rows = 0
    for fpath in csv_files:
        with open(fpath, 'r') as f:
            total_rows += sum(1 for _ in f) - 1  # минус заголовок

    logger.info(f"Total rows in all files: {total_rows}")
    sample_size = min(max_samples, total_rows)
    logger.info(f"Taking random sample of {sample_size} rows")

    # Reservoir sampling для выбора строк из множества файлов
    np.random.seed(random_state)
    chosen_indices = set(np.random.choice(total_rows, size=sample_size, replace=False))

    X_list, y_list = [], []
    current_offset = 0
    for fpath in csv_files:
        # Читаем файл чанками, чтобы не загружать целиком
        chunk_iter = pd.read_csv(fpath, chunksize=100_000)
        for chunk in chunk_iter:
            chunk_size = len(chunk)
            chunk_indices = np.arange(current_offset, current_offset + chunk_size)
            mask = np.isin(chunk_indices, list(chosen_indices))
            if mask.any():
                selected = chunk.iloc[mask]
                if "Label" in selected.columns:
                    y_list.append(selected["Label"].values.astype(int))
                    X_list.append(selected.drop(columns=["Label"]).values.astype(np.float32))
            current_offset += chunk_size

    X = np.vstack(X_list) if X_list else np.empty((0, 0))
    y = np.concatenate(y_list) if y_list else np.empty((0,))
    logger.info(f"Sample loaded: X shape {X.shape}, y shape {y.shape}")
    return X, y

def get_dataset_splits(force_reload: bool = False,
                       max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Возвращает X_train, X_test, y_train, y_test.
    Если max_samples задан, берёт подвыборку, иначе пытается загрузить все (риск MemoryError).
    """
    cache_dir = Path("data/splits")
    cache_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_sample{max_samples}" if max_samples else "_full"
    X_train_path = cache_dir / f"X_train{suffix}.npy"
    X_test_path = cache_dir / f"X_test{suffix}.npy"
    y_train_path = cache_dir / f"y_train{suffix}.npy"
    y_test_path = cache_dir / f"y_test{suffix}.npy"

    if not force_reload and all(p.exists() for p in [X_train_path, X_test_path, y_train_path, y_test_path]):
        logger.info("Loading cached splits...")
        X_train = np.load(X_train_path)
        X_test = np.load(X_test_path)
        y_train = np.load(y_train_path)
        y_test = np.load(y_test_path)
        return X_train, X_test, y_train, y_test

    logger.info("Loading data...")
    if max_samples is not None:
        X, y = load_sample_data(max_samples=max_samples)
    else:
        # Опасный путь – может упасть
        X, y = load_all_processed_data()  # оставлена для совместимости, но не рекомендуется
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    np.save(X_train_path, X_train)
    np.save(X_test_path, X_test)
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)
    logger.info("Splits saved to disk.")
    return X_train, X_test, y_train, y_test

def load_all_processed_data(data_dir: Path = Path("data/processed")):
    """Загружает все данные – используется только если памяти достаточно."""
    csv_files = sorted(data_dir.glob("*.csv"))
    dfs = [pd.read_csv(f) for f in csv_files]
    full_df = pd.concat(dfs, ignore_index=True)
    y = full_df["Label"].values.astype(int)
    X = full_df.drop(columns=["Label"]).values.astype(np.float32)
    return X, y

# ─────────────────────────────────────────────────────────────────────────────
# Генератор данных для Keras
# ─────────────────────────────────────────────────────────────────────────────

class CSVDataGenerator(tf.keras.utils.Sequence):
    """
    Генератор батчей, читающий данные из списка CSV-файлов по требованию.
    """
    def __init__(self, file_paths: List[Path], batch_size: int = 256,
                 shuffle: bool = True, seed: int = 42):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._file_row_counts = []
        self._total_rows = 0
        self._file_offsets = []  # глобальный индекс начала каждого файла
        self._build_index()
        self.on_epoch_end()

    def _build_index(self):
        """Подсчитывает количество строк в каждом файле и вычисляет смещения."""
        offset = 0
        for fpath in self.file_paths:
            with open(fpath, 'r', encoding='utf-8') as f:
                # Быстрый подсчёт строк (минус заголовок)
                n_rows = sum(1 for _ in f) - 1
            self._file_row_counts.append(n_rows)
            self._file_offsets.append(offset)
            offset += n_rows
        self._total_rows = offset

    def __len__(self):
        return int(np.ceil(self._total_rows / self.batch_size))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self._total_rows)
        batch_indices = self.indices[start_idx:end_idx]

        # Группируем запрошенные строки по файлам
        file_to_local_indices = {}
        for global_idx in batch_indices:
            # Бинарный поиск файла по глобальному индексу
            file_idx = np.searchsorted(self._file_offsets, global_idx, side='right') - 1
            if file_idx < 0:
                file_idx = 0
            local_idx = global_idx - self._file_offsets[file_idx]
            if file_idx not in file_to_local_indices:
                file_to_local_indices[file_idx] = []
            file_to_local_indices[file_idx].append(local_idx)

        X_batch, y_batch = [], []
        for file_idx, local_rows in file_to_local_indices.items():
            fpath = self.file_paths[file_idx]
            # Читаем только нужные строки (оптимизация: можно кэшировать DataFrame для файла)
            df = pd.read_csv(fpath)
            for r in local_rows:
                row = df.iloc[r]
                y_batch.append(row["Label"])
                X_batch.append(row.drop("Label").values.astype(np.float32))

        return np.array(X_batch), np.array(y_batch)

    def on_epoch_end(self):
        self.indices = np.arange(self._total_rows)
        if self.shuffle:
            np.random.shuffle(self.indices)