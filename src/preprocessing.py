import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(filepath: str = "data/raw/CarPrice_Assignment.csv"):
    """Загрузка данных из CSV."""
    df = pd.read_csv(filepath)
    print(f"Загружено {df.shape[0]} строк, {df.shape[1]} столбцов")
    return df


def clean_data(df):
    """
    Очистка данных:
    - Удаление дубликатов
    - Обработка пропусков
    - Удаление неинформативных колонок
    """
    df_clean = df.copy()

    # Удаляем дубликаты
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"Удалено дубликатов: {initial_rows - len(df_clean)}")

    # Проверяем пропуски
    missing = df_clean.isnull().sum()
    if missing.sum() > 0:
        print(f"Пропуски в колонках:\n{missing[missing > 0]}")
        # Для числовых - медиана, для категориальных - мода
        for col in df_clean.columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    else:
        print("Пропусков нет")

    # Удаляем неинформативные колонки
    cols_to_drop = ['car_ID', 'CarName']  # ID и полное название модели
    df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns])

    return df_clean


def encode_categorical(df):
    """
    Кодирование категориальных признаков.
    Бинарные → LabelEncoder, остальные → One-Hot Encoding.
    """
    df_encoded = df.copy()

    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()

    # Если 'price' в categorical_cols, убираем (это таргет)
    if 'price' in categorical_cols:
        categorical_cols.remove('price')

    # Бинарные колонки кодируем через LabelEncoder
    binary_cols = ['fueltype', 'aspiration', 'doornumber', 'enginelocation']
    for col in binary_cols:
        if col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            categorical_cols.remove(col)

    # Остальные категориальные — One-Hot Encoding
    if categorical_cols:
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)

    return df_encoded


def split_data(df, target_col='price', test_size=0.2, val_size=0.2, random_state=42):
    """
    Разделение данных на train/val/test.
    Сначала train (60%) и temp (40%), затем temp на val (50% от temp) и test (50% от temp).
    Итог: train 60%, val 20%, test 20%.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Сначала train (60%) и temp (40%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=random_state
    )

    # Делим temp на val и test (каждый по 20% от исходного)
    val_ratio = val_size / (val_size + test_size)  # = 0.5
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )

    print(f"Train: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """
    Стандартизация признаков (fit только на train).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler