"""
Предобработка датасета CSECIC-IDS2018 с обязательным StandardScaler.
Исправлено удаление колонок Flow ID, Src IP, Dst IP и др.
"""

import os
import re
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

COLUMNS_TO_DROP = [
    "id", "Flow ID", "Src IP", "Dst IP", "Timestamp", "Attempted Category"
]

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def parse_date_from_filename(filepath: Path) -> str:
    """Извлекает дату в формате YYYY-MM-DD для хронологической сортировки."""
    match = re.search(r"(\d{2})-(\d{2})-(\d{4})", filepath.name)
    if match:
        day, month, year = match.groups()
        return f"{year}-{month}-{day}"
    return filepath.name


def normalize_name(name: str) -> str:
    """Приводит имя столбца к единому формату (замена пробелов, тире, слешей на _)."""
    name = name.strip().replace(" ", "_").replace("/", "_").replace("-", "_")
    return re.sub(r"_+", "_", name)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка, удаление ненужных колонок, бинаризация меток, приведение типов.
    """
    # 1. Нормализация имён столбцов
    df.columns = [normalize_name(c) for c in df.columns]

    # 2. Удаление колонок из COLUMNS_TO_DROP
    normalized_drop = [normalize_name(c) for c in COLUMNS_TO_DROP]
    cols_to_drop = [c for c in normalized_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # 3. Бинаризация метки
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(str).str.strip().str.upper()
        df["Label"] = (df["Label"] != "BENIGN").astype(int)
    else:
        logger.warning("Колонка 'Label' не найдена в файле!")

    # 4. Приведение всех оставшихся колонок к числовому типу
    df = df.apply(pd.to_numeric, errors="coerce")

    # 5. Обработка бесконечностей и пропусков
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # 6. Гарантируем целочисленный тип метки
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(int)

    return df


def fit_scaler(raw_files: list) -> StandardScaler:
    """
    Считывает все CSV-файлы чанками и обучает StandardScaler на признаках (без Label).
    """
    scaler = StandardScaler()
    for raw_path in raw_files:
        logger.info(f"Обучение scaler на {raw_path.name}")
        for chunk in pd.read_csv(raw_path, chunksize=500_000, low_memory=False):
            chunk = clean_dataframe(chunk)
            if "Label" not in chunk.columns:
                continue
            X = chunk.drop(columns=["Label"])
            if X.empty:
                continue
            scaler.partial_fit(X)
    return scaler


def transform_and_save(raw_files: list, scaler: StandardScaler):
    """
    Читает чанками, масштабирует признаки и сохраняет результат в data/processed.
    """
    for raw_path in raw_files:
        processed_path = PROCESSED_DIR / raw_path.name
        logger.info(f"Масштабирование: {raw_path.name}")
        first_chunk = True
        for chunk in pd.read_csv(raw_path, chunksize=500_000, low_memory=False):
            chunk = clean_dataframe(chunk)
            if "Label" not in chunk.columns:
                continue
            y = chunk["Label"]
            X = chunk.drop(columns=["Label"])

            X_scaled = scaler.transform(X)
            scaled_chunk = pd.DataFrame(X_scaled, columns=X.columns)
            scaled_chunk["Label"] = y.values

            write_header = first_chunk
            scaled_chunk.to_csv(processed_path, mode='a', index=False, header=write_header)
            first_chunk = False
        logger.info(f"Сохранено в {processed_path}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(RAW_DIR.glob("*.csv"), key=parse_date_from_filename)
    if not raw_files:
        logger.error(f"Не найдено .csv файлов в {RAW_DIR}")
        return

    scaler_path = PROCESSED_DIR / "scaler.pkl"

    if scaler_path.exists():
        logger.info("Scaler уже найден, загрузка.")
        scaler = joblib.load(scaler_path)
    else:
        logger.info("Обучение StandardScaler на всех данных...")
        scaler = fit_scaler(raw_files)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler сохранён в {scaler_path}")

    logger.info("Применение масштабирования и сохранение файлов...")
    transform_and_save(raw_files, scaler)

    logger.info("Предобработка завершена.")


if __name__ == "__main__":
    main()