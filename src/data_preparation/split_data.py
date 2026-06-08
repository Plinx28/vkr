#!/usr/bin/env python3
"""
Разбиение масштабированных данных из data/processed на train/val/test со стратификацией (60/20/20).

Создаёт папки data/train, data/val, data/test и для каждого исходного файла
записывает три файла с постфиксами _train.csv, _val.csv, _test.csv.
"""

import gc
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SOURCE_DIR = Path("data/cut_features")
TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")
TEST_DIR = Path("data/test")
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def split_file(csv_path: Path) -> None:
    """
    Стратифицированно делит один CSV-файл на train/val/test
    и сохраняет результат в соответствующие папки.
    """
    logger.info(f"Обрабатываю {csv_path.name} ...")
    df = pd.read_csv(csv_path, sep=";")
    if "Label" not in df.columns:
        logger.warning(f"Колонка 'Label' не найдена в {csv_path.name}, пропускаю.")
        return

    y = df["Label"]
    X = df.drop(columns=["Label"])

    # 1. Отделить test (20%) стратифицированно
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    # 2. Из оставшихся 80% отделить validation (25% от temp -> 20% от исходного)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp
    )

    # Формируем итоговые DataFrame'ы
    train_df = X_train.copy()
    train_df["Label"] = y_train

    val_df = X_val.copy()
    val_df["Label"] = y_val

    test_df = X_test.copy()
    test_df["Label"] = y_test

    # Имя файла без расширения, для создания новых имён
    stem = csv_path.stem  # e.g. "02-03-2018"
    # Замена дефисов на подчеркивания для единообразия (опционально)
    safe_name = stem.replace("-", "_")

    # Сохранение
    train_df.to_csv(TRAIN_DIR / f"{safe_name}_train.csv", index=False)
    val_df.to_csv(VAL_DIR / f"{safe_name}_val.csv", index=False)
    test_df.to_csv(TEST_DIR / f"{safe_name}_test.csv", index=False)

    logger.info(
        f"Сохранено: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    # Явная очистка памяти
    del (
        df,
        X,
        y,
        X_temp,
        X_test,
        y_temp,
        y_test,
        X_train,
        X_val,
        y_train,
        y_val,
        train_df,
        val_df,
        test_df,
    )
    gc.collect()


def main():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # Получаем все CSV-файлы из data/processed (исключая scaler.pkl, если он там)
    csv_files = sorted(SOURCE_DIR.glob("*.csv"))
    if not csv_files:
        logger.error(f"В {SOURCE_DIR} нет CSV-файлов для разбиения.")
        return

    logger.info(f"Найдено {len(csv_files)} файлов для обработки.")
    for fpath in csv_files:
        try:
            split_file(fpath)
        except Exception as e:
            logger.error(f"Ошибка при обработке {fpath.name}: {e}")

    logger.info("Разбиение завершено.")


if __name__ == "__main__":
    main()
