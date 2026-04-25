"""
Модуль анализа распределения меток (Label) в данных сетевого трафика.
Универсально работает с любыми папками, содержащими CSV-файлы со столбцом 'Label'.
Строит столбчатую диаграмму с логарифмической шкалой для всех уникальных значений metок.
Сохраняет числовую сводку в CSV и график в PNG.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_labels_in_directory(data_dir: Path, output_prefix: str, figures_dir: Path, analysis_dir: Path):
    """
    Подсчитывает распределение всех уникальных значений столбца 'Label'
    во всех CSV-файлах из указанной директории.

    Сохраняет:
        - CSV с колонками 'Label' и 'Count' в analysis_dir / f"{output_prefix}_class_distribution.csv"
        - PNG с диаграммой в figures_dir / f"{output_prefix}_class_distribution.png"

    Параметры:
        data_dir      – путь к папке с CSV-файлами
        output_prefix – префикс для имени выходных файлов (например, 'raw' или 'processed')
        figures_dir   – путь к папке для сохранения графиков
        analysis_dir  – путь к папке для сохранения CSV-файлов
    """
    logger.info(f"Анализ меток в {data_dir} ...")
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(f"В папке {data_dir} нет CSV-файлов.")
        return

    all_labels = []
    for fpath in csv_files:
        try:
            # Загружаем только столбец Label для минимизации потребления памяти
            df = pd.read_csv(fpath, usecols=["Label"])
            # Приводим к строковому типу для единообразия, убираем пробелы по краям
            labels = df["Label"].astype(str).str.strip()
            all_labels.append(labels)
        except Exception as e:
            logger.error(f"Ошибка при чтении {fpath.name}: {e}")

    if not all_labels:
        logger.warning("Не удалось загрузить данные.")
        return

    # Объединяем все метки в одну серию
    all_labels_series = pd.concat(all_labels, ignore_index=True)
    # Подсчитываем частоты и сортируем по убыванию
    vc = all_labels_series.value_counts()

    # ── Сохранение CSV ──
    csv_path = analysis_dir / f"{output_prefix}_class_distribution.csv"
    vc_df = vc.reset_index()
    vc_df.columns = ["Label", "Count"]
    vc_df.to_csv(csv_path, index=False)
    logger.info(f"Сводка сохранена: {csv_path}")

    # ── Построение диаграммы ──
    labels = vc_df["Label"].tolist()
    counts = vc_df["Count"].tolist()

    # Цветовая схема: 'BENIGN' или '0' выделяем синим, остальное – оранжевым
    colors = []
    for lbl in labels:
        if lbl in ("BENIGN", "0", "benign"):
            colors.append("steelblue")
        else:
            colors.append("coral")

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.5), 6))
    bars = ax.bar(range(len(labels)), counts, color=colors)

    # Настройка оси X
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    # Логарифмическая шкала по Y
    ax.set_yscale("log")
    ax.set_ylabel("Количество записей")
    ax.set_xlabel("Метка класса")
    ax.set_title(f"Распределение меток в данных")

    # Добавление числовых подписей над столбцами
    for bar, count in zip(bars, counts):
        ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() * 1.05,
                f'{count:,}',
                ha='center',
                va='bottom',
                fontsize=7,
            )

    # Сетка по Y
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()

    # Сохранение графика
    fig_path = figures_dir / f"{output_prefix}_class_distribution.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info(f"График сохранён: {fig_path}")


def main():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    figures_dir = Path("reports/figures")
    analysis_dir = Path("reports/data_analysis")

    figures_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Сырые данные (raw) – строковые метки
    if raw_dir.exists():
        analyze_labels_in_directory(raw_dir, "raw", figures_dir, analysis_dir)
    else:
        logger.warning(f"Папка {raw_dir} не найдена, анализ raw пропущен.")

    # Обработанные данные (processed) – бинарные метки 0/1
    if processed_dir.exists():
        analyze_labels_in_directory(processed_dir, "processed", figures_dir, analysis_dir)
    else:
        logger.warning(f"Папка {processed_dir} не найдена, анализ processed пропущен.")

    logger.info("Анализ распределения меток завершён.")


if __name__ == "__main__":
    main()