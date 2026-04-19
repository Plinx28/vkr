"""
Предобработка датасета CSECIC-IDS2018 для последующего обучения моделей.
"""

import os
import re
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Колонки, которые не несут смысла для ML (идентификаторы, IP, время, мета-категории)
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


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка, кодировка меток и приведение типов."""
    
    # 1. Убираем пробелы и спецсимволы в названиях колонок
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)
    df.columns = [re.sub(r"_+", "_", c) for c in df.columns]

    # 2. Формируем список колонок для удаления (с учётом изменённых имён)
    col_map = {
        orig: orig.strip().replace(" ", "_").replace("/", "_").replace("-", "_")
        for orig in COLUMNS_TO_DROP
    }
    cols_to_drop = [col_map[c] for c in col_map if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # 3. Кодировка метки: BENIGN -> 0, всё остальное -> 1
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(str).str.strip().str.upper()
        df["Label"] = (df["Label"] != "BENIGN").astype(int)
    else:
        logger.warning("Колонка 'Label' не найдена в файле!")

    # 4. Приводим все оставшиеся колонки к числовому типу
    df = df.apply(pd.to_numeric, errors="coerce")

    # 5. Обработка бесконечностей и пропусков (типично для сетевых фичей)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)  # Можно заменить на median/forward_fill при необходимости

    # 6. Гарантируем целочисленный тип для метки
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(int)

    return df


def main():
    # Создаём директории, если их нет
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Сортируем файлы хронологически по имени
    raw_files = sorted(RAW_DIR.glob("*.csv"), key=parse_date_from_filename)
    if not raw_files:
        logger.error(f"Не найдено .csv файлов в {RAW_DIR}")
        return

    logger.info(f"Найдено {len(raw_files)} файлов. Начинаю обработку...")

    for raw_path in raw_files:
        processed_path = PROCESSED_DIR / raw_path.name
        logger.info(f"Обработка: {raw_path.name}")
        try:
            # low_memory=False подавляет предупреждения о смешанных типах при чтении
            df = pd.read_csv(raw_path, low_memory=False)
            df_processed = clean_dataframe(df)
            df_processed.to_csv(processed_path, index=False)
            logger.info(f"Сохранено: {processed_path.name} | Форма: {df_processed.shape}")
        except Exception as e:
            logger.error(f"Ошибка при обработке {raw_path.name}: {e}")

    logger.info("Подготовка данных завершена.")


if __name__ == "__main__":
    main()