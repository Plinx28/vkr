"""
Очистка уже разбитых CSV-файлов в data/train, data/val, data/test:
- удаление ненужных колонок (IP, Flow ID, Timestamp и др.)
- бинаризация метки (BENIGN → 0, остальное → 1)
- приведение всех значений к числовому типу
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_DIRS = [Path("data/train"), Path("data/val"), Path("data/test")]

CHUNK_SIZE = 500_000
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def normalize_name(name: str) -> str:
    """Приводит имя столбца к единому формату."""
    name = name.strip().replace(" ", "_").replace("/", "_").replace("-", "_")
    return re.sub(r"_+", "_", name)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Применяет все шаги очистки к одному DataFrame."""
    # 1. Нормализация имён столбцов
    df.columns = [normalize_name(c) for c in df.columns]

    # 2. Бинаризация метки
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(str).str.strip().str.upper()
        df["Label"] = (df["Label"] != "BENIGN").astype(int)
    else:
        logger.warning("Колонка 'Label' не найдена – пропускаем бинаризацию")

    # 3. Преобразование всех колонок в числовой тип (несовместимое станет NaN)
    df = df.apply(pd.to_numeric)

    # 4. Замена бесконечностей и пропусков
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # 5. Гарантируем целочисленный тип метки (если она есть)
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(int)

    return df


def clean_file(file_path: Path) -> None:
    """Очищает CSV-файл и перезаписывает его."""
    logger.info(f"Очистка {file_path.name} ...")
    tmp_path = file_path.with_suffix(".tmp")

    try:
        first_chunk = True
        with pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False) as reader:
            for chunk in reader:
                cleaned = clean_dataframe(chunk)
                # Первый чанк записываем с заголовком, последующие – без
                cleaned.to_csv(tmp_path, mode="a", index=False, header=first_chunk)
                first_chunk = False

        # Замена исходного файла очищенным
        tmp_path.replace(file_path)
    except Exception as e:
        logger.error(f"Ошибка при обработке {file_path.name}: {e}")
        if tmp_path.exists():
            tmp_path.unlink()  # удаляем временный файл, если что-то пошло не так


def main():
    for target_dir in TARGET_DIRS:
        if not target_dir.exists():
            logger.warning(f"Папка {target_dir} не найдена, пропускаю.")
            continue
        csv_files = sorted(target_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"В папке {target_dir} нет CSV-файлов.")
            continue
        logger.info(f"=== Обработка {len(csv_files)} файлов в {target_dir} ===")
        for fpath in csv_files[:2]:
            clean_file(fpath)
    logger.info("Очистка завершена.")


if __name__ == "__main__":
    main()
