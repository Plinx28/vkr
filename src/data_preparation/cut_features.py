import glob
import os

import pandas as pd

RAW_DIR = "data/raw"
OUT_DIR = "data/cut_features"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_FEATURES = [
    "Flow_Duration",
    "Total_Fwd_Packet",
    "Total_Bwd_packets",
    "Fwd_Packet_Length_Mean",
    "Fwd_Packet_Length_Std",
    "Fwd_IAT_Mean",
    "Bwd_IAT_Mean",
    "Flow_Bytes_s",
    "Flow_Packets_s",
    "FIN_Flag_Cnt",
    "SYN_Flag_Cnt",
    "RST_Flag_Cnt",
    "ACK_Flag_Count",
    "Active_Mean",
    "Idle_Mean",
    "FWD_Init_Win_Bytes",
    "Label"
]


def normalize_name(name):
    """Приводит имя столбца к единому формату для сравнения (нижний регистр, замена пробелов и спецсимволов на '_')"""
    return (
        str(name).strip().lower()
        .replace(' ', '_')
        .replace('/', '_')
        .replace('(', '')
        .replace(')', '')
        .replace('.','_')
        )

def process_csv(input_path, output_path):
    """Читает CSV, нормализует заголовки, оставляет только нужные признаки и сохраняет."""
    try:
        # Попытка определить разделитель
        sep = ","
        df = pd.read_csv(input_path, sep=sep, on_bad_lines="skip", encoding="utf-8-sig", nrows=2)
        if len(df.columns) <= 2:
            sep = ";"
            df = pd.read_csv(input_path, sep=sep, on_bad_lines="skip", encoding="utf-8-sig")

        # Полное чтение файла с верным разделителем
        df = pd.read_csv(input_path, sep=sep, on_bad_lines="skip", encoding="utf-8-sig")

        # 1. Нормализация и сопоставление имен столбцов
        # Создаем карту: {нормализованное_имя_из_файла: целевое_имя_из_списка}
        target_normalized = {normalize_name(f): f for f in TARGET_FEATURES}

        rename_map = {}
        for col in df.columns:
            norm_col = normalize_name(col)
            if norm_col in target_normalized:
                rename_map[col] = target_normalized[norm_col]

        # 2. Переименование найденных столбцов
        df = df.rename(columns=rename_map)

        # 3. Фильтрация: оставить только те столбцы, которые есть в TARGET_FEATURES
        cols_to_keep = [col for col in TARGET_FEATURES if col in df.columns]

        if not cols_to_keep:
            print(f"Ни один признак не найден в {os.path.basename(input_path)}. Пропуск.")
            return 0

        missing = set(TARGET_FEATURES) - set(cols_to_keep)
        if missing:
            print(f"В {os.path.basename(input_path)} отсутствуют: {list(missing)}")

        df_cut = df[cols_to_keep].copy()
        df_cut["Label"] = (df_cut["Label"] != "BENIGN").astype(int)

        # 4. Сохранение
        df_cut.to_csv(output_path, index=False, sep=";")
        return len(df_cut)

    except Exception as e:
        print(f"Ошибка обработки {os.path.basename(input_path)}: {e}")
        return 0


# ================= ОСНОВНОЙ ЦИКЛ =================
print("Поиск CSV файлов в data/raw...")
csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))

if not csv_files:
    print("Файлы .csv не найдены в папке data/raw. Проверьте путь.")
else:
    print(f"Найдено файлов: {len(csv_files)}\n")
    total_rows = 0
    debug_done = False

    for i, filepath in enumerate(csv_files, 1):
        filename = os.path.basename(filepath)
        out_path = os.path.join(OUT_DIR, f"cut_{filename}")

        print(f"[{i}/{len(csv_files)}] Обработка: {filename}")
        rows = process_csv(filepath, out_path)
        total_rows += rows

        # Вывод реальных заголовков первого файла для верификации
        if i == 1 and not debug_done:
            try:
                sample = pd.read_csv(filepath, sep=";", nrows=0, encoding="utf-8-sig")
                if len(sample.columns) <= 2:
                    sample = pd.read_csv(filepath, sep=",", nrows=0, encoding="utf-8-sig")

                print(f"\nИсходные заголовки в первом файле ({filename}):")
                print(list(sample.columns))
                debug_done = True
            except Exception as e:
                print(f"Не удалось прочитать заголовки для отладки: {e}")

    print(f"\nВсего обработано строк: {total_rows}")
    print(f"Результаты сохранены в: {os.path.abspath(OUT_DIR)}")
