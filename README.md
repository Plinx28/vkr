# vkr
ВКР

```
project_root/
│
├── data/                      # Исходные и обработанные данные
│   ├── raw/                   # Разархивированные .csv файлы CSECICIDS2018_improved.zip
│   ├── processed/             # Очищенные и нормализованные наборы данных (список .CSV файлов)
│   └── splits/                # Индексы обучающей и тестовой выборок (train_idx.npy, test_idx.npy)
│
├── models/                    # Сохранённые артефакты обученных моделей
│   ├── logistic_regression/
│   ├── xgboost/
│   ├── mlp/
│   └── autoencoder/
│
├── reports/                   # Результаты и отчётность
│   ├── figures/               # Графики истории обучения (loss, accuracy), ROC-кривые
│   ├── metrics/               # Таблицы с итоговыми метриками (.csv)
│   └── training_logs/         # Логи TensorBoard или текстовые логи обучения
│
├── src/                       # Исходный код
│   ├── __init__.py
│   ├── data_preparation.py    # Модуль загрузки, очистки, разделения и сохранения данных
│   ├── utils.py               # Вспомогательные функции (установка seed, таймеры, метрики)
│   ├── models/                # Пакет с реализациями моделей
│   │   ├── __init__.py
│   │   ├── base_model.py      # Абстрактный базовый класс с интерфейсом fit/predict/save/load
│   │   ├── logistic_regression.py
│   │   ├── xgboost_model.py
│   │   ├── mlp_model.py
│   │   └── autoencoder_model.py
│   ├── train.py               # Скрипт запуска обучения одной модели (принимает аргументы командной строки)
│   ├── evaluate.py            # Скрипт вычисления метрик на тестовой выборке и построения графиков
│   └── run_all_experiments.py # Оркестратор для последовательного запуска всех моделей
│
│
├── requirements.txt           # Зависимости Python
└── README.md                  # Инструкция по воспроизведению экспериментов
```
TensorBoard для визуализации обучения нейросетей запускается командой:

```tensorboard --logdir reports/training_logs```
# Подготовка данных
<p>Сначала выполняется масштабирование признаков и сохранение данных в data/processed</p>
`python src/data_preparation.py`

<p>Затем данные разделяются на выборки train, val, test</p>
`python src/split_data.py`

# Запуск обучения

`python src/train.py --model mlp --params configs/mlp_params.json`
`python src/train.py --model mlp --max_samples 20000000 --params configs/mlp_params.json`

# Тестирование обученных моделей

`python src/evaluate.py --model all`
