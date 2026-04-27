"""
Вспомогательные функции: фиксация seed, таймеры, метрики, генератор без перемешивания,
callback для метрик валидации, подбор порога.
"""

import os
import time
import random
import logging
from pathlib import Path
from typing import Optional, Dict, List
from contextlib import contextmanager

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    precision_recall_curve
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Seed
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
# Генератор
# ─────────────────────────────────────────────────────────────────────────────

class CSVDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths: List[Path], batch_size: int = 256,
                 shuffle: bool = False, seed: int = 42):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._file_row_counts = []
        self._total_rows = 0
        self._file_offsets = []
        self._build_index()
        self.on_epoch_end()

    def _build_index(self):
        offset = 0
        for fpath in self.file_paths:
            with open(fpath, 'r', encoding='utf-8') as f:
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

        file_to_local = {}
        for gidx in batch_indices:
            fidx = np.searchsorted(self._file_offsets, gidx, side='right') - 1
            if fidx < 0:
                fidx = 0
            lidx = gidx - self._file_offsets[fidx]
            file_to_local.setdefault(fidx, []).append(lidx)

        X_batch, y_batch = [], []
        for fidx, lids in file_to_local.items():
            df = pd.read_csv(self.file_paths[fidx])
            for r in lids:
                row = df.iloc[r]
                y_batch.append(row["Label"])
                X_batch.append(row.drop("Label").values.astype(np.float32))
        return np.array(X_batch), np.array(y_batch)

    def on_epoch_end(self):
        self.indices = np.arange(self._total_rows)
        if self.shuffle:
            np.random.shuffle(self.indices)

# ─────────────────────────────────────────────────────────────────────────────
# Callback метрик на валидации
# ─────────────────────────────────────────────────────────────────────────────

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, threshold=0.5):
        super().__init__()
        self.validation_data = validation_data
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if isinstance(self.validation_data, tf.keras.utils.Sequence):
            y_true, y_pred = [], []
            for Xb, yb in self.validation_data:
                proba = self.model.predict(Xb, verbose=0)
                pred = (proba > self.threshold).astype(int).flatten()
                y_true.extend(yb)
                y_pred.extend(pred)
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
        else:
            X_val, y_val = self.validation_data
            proba = self.model.predict(X_val, verbose=0)
            y_pred = (proba > self.threshold).astype(int).flatten()
            y_true = y_val

        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        logs['val_precision'] = p
        logs['val_recall'] = r
        logs['val_f1'] = f1
        logs['val_mcc'] = mcc
        print(f" - val_precision: {p:.4f} - val_recall: {r:.4f} - val_f1: {f1:.4f} - val_mcc: {mcc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Подбор порога
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.0)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1s)
    return thresholds[best_idx]

# ─────────────────────────────────────────────────────────────────────────────
# Загрузка подвыборки (для классических моделей)
# ─────────────────────────────────────────────────────────────────────────────

def load_sample_data(data_dir=Path("data/processed"), max_samples=1_000_000, random_state=42):
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")
    total = 0
    for f in csv_files:
        with open(f, 'r') as fp:
            total += sum(1 for _ in fp) - 1
    sample_size = min(max_samples, total)
    np.random.seed(random_state)
    chosen = set(np.random.choice(total, size=sample_size, replace=False))
    X_list, y_list = [], []
    cur = 0
    for f in csv_files:
        for chunk in pd.read_csv(f, chunksize=100_000):
            idx = np.arange(cur, cur + len(chunk))
            mask = np.isin(idx, list(chosen))
            if mask.any():
                sel = chunk.iloc[mask]
                y_list.append(sel["Label"].values.astype(int))
                X_list.append(sel.drop(columns=["Label"]).values.astype(np.float32))
            cur += len(chunk)
    X = np.vstack(X_list) if X_list else np.empty((0,0))
    y = np.concatenate(y_list) if y_list else np.empty((0,))
    return X, y

def get_dataset_splits(force_reload=False, max_samples=None):
    cache_dir = Path("data/splits")
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_sample{max_samples}" if max_samples else "_full"
    X_tr_path = cache_dir / f"X_train{suffix}.npy"
    X_te_path = cache_dir / f"X_test{suffix}.npy"
    y_tr_path = cache_dir / f"y_train{suffix}.npy"
    y_te_path = cache_dir / f"y_test{suffix}.npy"

    if not force_reload and all(p.exists() for p in [X_tr_path, X_te_path, y_tr_path, y_te_path]):
        logger.info("Loading cached splits...")
        return (np.load(X_tr_path), np.load(X_te_path),
                np.load(y_tr_path), np.load(y_te_path))

    if max_samples is not None:
        X, y = load_sample_data(max_samples=max_samples)
    else:
        csv_files = sorted(Path("data/processed").glob("*.csv"))
        dfs = [pd.read_csv(f) for f in csv_files]
        full = pd.concat(dfs, ignore_index=True)
        y = full["Label"].values.astype(int)
        X = full.drop(columns=["Label"]).values.astype(np.float32)
        del full

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    np.save(X_tr_path, X_train)
    np.save(X_te_path, X_test)
    np.save(y_tr_path, y_train)
    np.save(y_te_path, y_test)
    logger.info("Splits saved.")
    return X_train, X_test, y_train, y_test
