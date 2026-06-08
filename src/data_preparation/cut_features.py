import glob
import os

import pandas as pd

# ================= НАСТРОЙКИ =================
RAW_DIR = "data/raw"
OUT_DIR = "data/cut_features"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_MAPPING = {
    "Flow Duration": ["Flow Duration", "Flow Duration (ms)"],
    "Total Fwd Packet": ["Total Fwd Packet", "Total Fwd Packets", "Total_Fwd_Packets"],
    "Total Bwd packets": [
        "Total Bwd packets",
        "Total Bwd Packets",
        "Total_Bwd_Packets",
    ],

    "Fwd Packet Length Mean": [
        "Fwd Packet Length Mean",
        "Fwd Pkt Len Mean",
        "Fwd Packet Length Avg",
    ],
    "Fwd Packet Length Std": ["Fwd Packet Length Std", "Fwd Pkt Len Std"],

    "Fwd IAT Mean": ["Fwd IAT Mean", "Fwd IAT Avg"],
    "Bwd IAT Mean": ["Bwd IAT Mean", "Bwd IAT Avg"],

    "Flow Bytes/s": ["Flow Bytes/s", "Flow Byts/s", "Flow B/s"],
    "Flow Packets/s": ["Flow Packets/s", "Flow Pkts/s", "Flow P/s"],

    "FIN Flag Cnt": ["FIN Flag Cnt", "FIN Flag Count"],
    "SYN Flag Cnt": ["SYN Flag Cnt", "SYN Flag Count"],
    "RST Flag Cnt": ["RST Flag Cnt", "RST Flag Count"],
    "ACK Flag Count": ["ACK Flag Cnt", "ACK Flag Count"],

    "Active Mean": ["Active Mean", "Active Avg"],
    "Idle Mean": ["Idle Mean", "Idle Avg"],
    "FWD Init Win Bytes": [
        "FWD Init Win Bytes",
        "Fwd Init Win Byts",
        "Fwd Init Win Bytes",
    ],

    "Label": ["Label", " label", "LABEL", "Label "],
}


def process_csv(input_path, output_path, mapping):
    """Читает, сопоставляет имена, обрезает, преобразует метку и сохраняет."""
    try:
        sep = ","
        df = pd.read_csv(
            input_path,
            sep=sep,
            on_bad_lines="skip",
            encoding="utf-8-sig",
            nrows=2,
        )
        if len(df.columns) > 5:
            df = pd.read_csv(
                input_path, sep=sep, on_bad_lines="skip", encoding="utf-8-sig"
            )
        else:
            raise ValueError("Не удалось определить разделитель файла")

        # 1. Очистка заголовков
        df.columns = df.columns.str.strip()

        # 2. Сопоставление имен столбцов
        actual_to_standard = {}
        missing_features = []

        for standard, variants in mapping.items():
            found_col = None
            for v in variants:
                if v in df.columns:
                    found_col = v
                    break
            if found_col:
                actual_to_standard[found_col] = standard
            else:
                missing_features.append(standard)

        if missing_features:
            print(f"Отсутствуют признаки: {missing_features}")

        if not actual_to_standard:
            print("Ни один признак не найден. Пропуск файла.")
            return 0

        # 3. Выбор и переименование
        cols_to_keep = list(actual_to_standard.keys())
        df_cut = df[cols_to_keep].copy()
        df_cut.rename(columns=actual_to_standard, inplace=True)

        # 4. Сохранение
        df_cut.to_csv(output_path, index=False, sep=";")
        return len(df_cut)

    except Exception as e:
        print(f"Ошибка: {e}")
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
        rows = process_csv(filepath, out_path, FEATURE_MAPPING)
        total_rows += rows

        # Вывод реальных заголовков первого файла для верификации
        if i == 1 and not debug_done:
            try:
                sample = pd.read_csv(filepath, sep=";", nrows=0, encoding="utf-8-sig")
                sample.columns = sample.columns.str.strip()
                print(f"\n🔎 Реальные заголовки в файле {filename}:")
                print(list(sample.columns))
                debug_done = True
            except Exception:
                pass

    print(f"Всего обработано строк: {total_rows}")
    print(f"Результаты сохранены в: {os.path.abspath(OUT_DIR)}")
