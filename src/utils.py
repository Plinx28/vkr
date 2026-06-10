"""
Вспомогательные функции: фиксация seed, таймеры, метрики, загрузка данных,
callback метрик валидации, подбор порога.

Модуль содержит переиспользуемые утилиты, общие для скриптов обучения и оценки:
обеспечение воспроизводимости экспериментов, измерение времени выполнения,
расчёт метрик качества бинарной классификации и загрузку датасета из набора
CSV-файлов.
"""

import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    matthews_corrcoef,
)

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Фиксирует начальные значения генераторов случайных чисел.

    Устанавливает seed для модулей `random`, ``numpy` и `tensorflow`, а
    также переменную окружения `PYTHONHASHSEED`.

    Args:
        seed: Значение seed для всех генераторов случайных чисел.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)


@contextmanager
def timer(name: str = "Operation", log_level: int = logging.INFO):
    """Контекстный менеджер для замера времени выполнения блока кода.

    Логирует начало операции и её длительность (в секундах) при выходе из
    блока `with`, в том числе при возникновении исключения.

    Args:
        name: Название операции, выводимое в лог.
        log_level: Уровень логирования (например, `logging.INFO`).

    Yields:
        None: Управление передаётся в тело блока ``with``.
    """
    start = time.perf_counter()
    logger.log(log_level, f"{name} started...")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.log(log_level, f"{name} completed in {elapsed:.3f} seconds")


class Timer:
    """Объектный таймер для измерения интервалов времени.

    Альтернатива контекстному менеджеру :func:`timer` для случаев, когда нужно
    запускать и останавливать замер вручную.

    Attributes:
        start_time (Optional[float]): Момент старта (`perf_counter`) либо `None`.
        end_time (Optional[float]): Момент остановки (`perf_counter`) либо `None`.
    """
    def __init__(self):
        """Создаёт таймер без запуска отсчёта."""
        self.start_time = None
        self.end_time = None

    def start(self):
        """Запускает (или перезапускает) отсчёт времени.

        Returns:
            Timer: Текущий экземпляр таймера (для цепочки вызовов).
        """
        self.start_time = time.perf_counter()
        return self

    def stop(self):
        """Останавливает отсчёт времени.

        Returns:
            float: Прошедшее с момента старта время в секундах.
        """
        self.end_time = time.perf_counter()
        return self.elapsed

    @property
    def elapsed(self) -> float:
        """Прошедшее время в секундах.

        Если таймер не был запущен, возвращает 0. Если таймер запущен,
        но не остановлен, возвращает время с момента старта до текущего момента.

        Returns:
            float: Длительность интервала в секундах.
        """
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return end - self.start_time


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Вычисляет основные метрики качества классификации.

    Рассчитывает precision, recall, F1-меру и коэффициент корреляции Мэтьюса
    (MCC) по предсказанным меткам. Если переданы вероятности, дополнительно
    вычисляется PR-AUC (площадь под кривой precision-recall).

    Args:
        y_true: Вектор истинных меток формы (n_samples,).
        y_pred: Вектор предсказанных меток формы (n_samples,).
        y_proba: Необязательный вектор вероятностей положительного класса;
            при наличии используется для расчёта pr_auc.

    Returns:
        Словарь с метриками: precision, recall, f1, mcc и,
        если передан y_proba, pr_auc.
    """
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_proba is not None:
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """Выводит словарь метрик в консоль в читаемом виде.

    Args:
        metrics: Словарь «имя метрики → значение».
        title: Заголовок, печатаемый перед списком метрик.
    """
    print(f"\n{title}")
    print("-" * 40)
    for name, value in metrics.items():
        print(f"{name:12s}: {value:.4f}")


def load_data_from_dir(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Загружает все CSV-файлы из указанной папки в один массив X и y.

    Читает по очереди все файлы ``*.csv`` из директории, объединяет их в один
    DataFrame, после чего разделяет на матрицу признаков ``X`` и вектор меток
    ``y``. Целевая переменная ожидается в колонке ``"Label"``.

    Args:
        data_dir: Путь к директории с CSV-файлами.

    Returns:
        Кортеж ``(X, y)``, где ``X`` — матрица признаков типа ``float32``
        формы ``(n_samples, n_features)``, а ``y`` — вектор целочисленных
        меток формы ``(n_samples,)``.

    Raises:
        FileNotFoundError: Если в директории нет ни одного CSV-файла.
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
