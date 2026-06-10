"""
Модуль анализа распределения меток (Label) в данных сетевого трафика.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_labels_in_directory(
    data_dir: Path, output_prefix: str, figures_dir: Path, analysis_dir: Path
):
    """
    Подсчитывает распределение всех уникальных значений столбца 'Label'
    во всех CSV-файлах из указанной директории.

    Сохраняет:
        - CSV с колонками 'Label' и 'Count' в analysis_dir / f"{output_prefix}_class_distribution.csv"
        - PNG с диаграммой в figures_dir / f"{output_prefix}_class_distribution.png"

    Параметры:
        data_dir      – путь к папке с CSV-файлами
        output_prefix – префикс для имени выходных файлов (например, 'train', 'val', 'test', 'raw', 'processed')
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
            # Загрузка только столбца Label для минимизации потребления памяти
            df = pd.read_csv(fpath, usecols=["Label"])
            # Приведение к строковому типу для единообразия, убираем пробелы по краям
            labels = df["Label"].astype(str).str.strip()
            all_labels.append(labels)
        except Exception as e:
            logger.error(f"Ошибка при чтении {fpath.name}: {e}")

    if not all_labels:
        logger.warning("Не удалось загрузить данные.")
        return

    # Объединение всех меток в одну серию
    all_labels_series = pd.concat(all_labels, ignore_index=True)
    # Подсчет частоты и сортировка по убыванию
    vc = all_labels_series.value_counts()

    # Сохранение CSV
    csv_path = analysis_dir / f"{output_prefix}_class_distribution.csv"
    vc_df = vc.reset_index()
    vc_df.columns = ["Label", "Count"]
    vc_df.to_csv(csv_path, index=False)
    logger.info(f"Сводка сохранена: {csv_path}")

    # Построение диаграммы
    labels = vc_df["Label"].tolist()
    counts = vc_df["Count"].tolist()

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
    ax.set_title(f"Распределение меток в данных ({output_prefix})")

    # Добавление числовых подписей над столбцами
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.05,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # Сетка по Y
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()

    # Сохранение графика
    fig_path = figures_dir / f"{output_prefix}_class_distribution.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info(f"График сохранён: {fig_path}")


def main():
    # Каталоги для анализа
    raw_dir = Path("data/raw")
    train_dir = Path("data/train")
    val_dir = Path("data/val")
    test_dir = Path("data/test")

    figures_dir = Path("reports/figures")
    analysis_dir = Path("reports/data_analysis")

    figures_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Сырые данные (raw) – строковые метки
    if raw_dir.exists():
        analyze_labels_in_directory(raw_dir, "raw", figures_dir, analysis_dir)
    else:
        logger.warning(f"Папка {raw_dir} не найдена, анализ raw пропущен.")

    # Данные после разделения train/val/test
    for subset_name, subset_dir in [
        ("train", train_dir),
        ("val", val_dir),
        ("test", test_dir),
    ]:
        if subset_dir.exists():
            analyze_labels_in_directory(
                subset_dir, subset_name, figures_dir, analysis_dir
            )
        else:
            logger.warning(
                f"Папка {subset_dir} не найдена, анализ {subset_name} пропущен."
            )

    logger.info("Анализ распределения меток завершён.")


if __name__ == "__main__":
    main()
