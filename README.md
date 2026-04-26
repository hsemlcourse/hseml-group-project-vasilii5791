[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)

# ML Project — [Название проекта]

**Студент:** [ФИО / Student ID]

**Группа:** [Группа]

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Отчёт](#отчёт)

## Описание задачи

<!-- Кратко опишите задачу: что предсказываем, какой датасет, метрика качества -->

**Задача:** [Классификация / Регрессия / Кластеризация / ...]

**Датасет:** [Название и источник датасета]

**Целевая метрика:** [Accuracy / F1 / RMSE / ...]

## Структура репозитория

Опишите структуру проекта, сохранив при этом верхнеуровневые папки. Можно добавить новые при необходимости.

```
.
├── data
│ ├── processed # Очищенные данные (будет позже)
│ └── raw # Исходные файлы
│ └── CarPrice_Assignment.csv
├── models # Сохранённые модели (будет позже)
├── notebooks
│ ├── 01_eda.ipynb # EDA
│ └── 02_baseline.ipynb # Baseline-модель
├── presentation # Презентация для защиты
├── report
│ ├── images # Изображения для отчёта
│ └── report.md # Финальный отчёт
├── src
│ ├── init.py
│ ├── preprocessing.py # Предобработка данных
│ └── utils.py # Seed и метрики
├── tests # Тесты (будет позже)
├── requirements.txt
└── README.md
```

## Запуск

```bash
# 1. Клонировать репозиторий
git clone <url>
cd <repo-name>

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Запустить Jupyter
jupyter notebook notebooks/
```

## Данные

data/raw/CarPrice_Assignment.csv — исходные данные (205 строк × 26 колонок)

## Результаты Baseline модели (CP1)

### Метрики качества

| Модель                    | MAE ($)  | RMSE ($) | R²     |
| ------------------------- | -------- | -------- | ------ |
| Linear Regression (Train) | 1,230.65 | 1,644.35 | 0.9592 |
| Linear Regression (Val)   | 1,784.50 | 2,479.67 | 0.8314 |
| Linear Regression (Test)  | 2,134.25 | 3,231.68 | 0.8648 |

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
