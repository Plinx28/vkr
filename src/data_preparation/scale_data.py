#!/usr/bin/env python3
"""
Масштабирование уже разбитых данных с помощью StandardScaler.
Scaler обучается только на data/train, применяется ко всем трём наборам.
"""

import logging
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# ──────────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ──────────────────────────────────────────────────────────────────
TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")
TEST_DIR = Path("data/test")
SCALER_PATH = Path("data/scaler.pkl")

CHUNK_SIZE = 500_000  # размер чанка для чтения CSV
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def fit_scaler_on_train(train_dir: Path) -> StandardScaler:
    """
    Постепенно (чанками) читает все CSV из train_dir, извлекает признаки
    (все столбцы, кроме 'Label') и обучает StandardScaler.
    """
    scaler = StandardScaler()
    csv_files = sorted(train_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Нет CSV-файлов в {train_dir}")

    logger.info(f"Начинаю обучение scaler на {len(csv_files)} файлах из {train_dir}")
    for fpath in csv_files:
        logger.info(f"Обрабатываю {fpath.name}")
        for chunk in pd.read_csv(fpath, chunksize=CHUNK_SIZE):
            if "Label" not in chunk.columns:
                logger.warning(f"В {fpath.name} нет столбца 'Label', пропускаю чанк")
                continue
            X = chunk.drop(columns=["Label"])
            if X.empty:
                continue
            # Все столбцы уже должны быть числовыми (после очистки)
            scaler.partial_fit(X)
    logger.info("Обучение scaler завершено")
    return scaler


def transform_and_save(data_dir: Path, scaler: StandardScaler) -> None:
    """
    Применяет масштабирование ко всем CSV в data_dir и перезаписывает их.
    """
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(f"В папке {data_dir} нет CSV-файлов")
        return

    for fpath in csv_files:
        logger.info(f"Масштабирую {fpath.name}")
        output_chunks = []
        for chunk in pd.read_csv(fpath, chunksize=CHUNK_SIZE):
            if "Label" not in chunk.columns:
                logger.warning(f"В {fpath.name} нет Label – пропускаю чанк")
                continue
            y = chunk["Label"]
            X = chunk.drop(columns=["Label"])
            X_scaled = scaler.transform(X)
            scaled_chunk = pd.DataFrame(X_scaled, columns=X.columns)
            scaled_chunk["Label"] = y.values
            output_chunks.append(scaled_chunk)

        if not output_chunks:
            continue

        # Собираем все масштабированные чанки и перезаписываем файл
        full_df = pd.concat(output_chunks, ignore_index=True)
        full_df.to_csv(fpath, index=False)
        logger.info(f"Файл {fpath.name} обновлён")


def main():
    # Проверяем наличие папок
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if not d.exists():
            logger.error(f"Папка {d} не найдена. Сначала выполните разбиение данных.")
            return

    # 1. Обучаем scaler (или загружаем, если уже есть)
    if SCALER_PATH.exists():
        logger.info("Загружаю существующий scaler")
        scaler = joblib.load(SCALER_PATH)
    else:
        scaler = fit_scaler_on_train(TRAIN_DIR)
        joblib.dump(scaler, SCALER_PATH)
        logger.info(f"Scaler сохранён в {SCALER_PATH}")

    # 2. Применяем scaler к train, val, test
    for subset_dir, subset_name in [
        (TRAIN_DIR, "train"),
        (VAL_DIR, "val"),
        (TEST_DIR, "test"),
    ]:
        logger.info(f"=== Обработка {subset_name} ===")
        transform_and_save(subset_dir, scaler)

    logger.info("Масштабирование завершено. Все наборы обновлены.")


if __name__ == "__main__":
    main()
