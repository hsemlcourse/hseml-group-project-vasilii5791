import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(filepath: str = "data/raw/Cars Datasets 2025.csv") -> pd.DataFrame:
    """Загрузка данных из CSV с автоопределением кодировки."""
    encodings_to_try = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"Загружено {df.shape[0]} строк, {df.shape[1]} столбцов (кодировка: {enc})")
            print(f"Колонки: {df.columns.tolist()}")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError("Не удалось прочитать CSV ни с одной из кодировок")


def clean_numeric_value(value):
    """Очистка числового значения: убирает $, запятые, единицы измерения, берёт среднее для диапазонов."""
    if pd.isna(value):
        return np.nan

    value_str = str(value).strip()

    # Убираем $ и запятые в числах
    value_str = value_str.replace('$', '').replace(',', '')

    # Убираем единицы измерения
    value_str = value_str.replace(' cc', '').replace(' hp', '').replace(' Nm', '')
    value_str = value_str.replace(' km/h', '').replace(' sec', '').replace(' cc', '')

    # Обработка диапазонов (например, "70-85", "100 - 140", "$12,000-$15,000")
    if '-' in value_str:
        parts = value_str.split('-')
        try:
            nums = [float(p.strip()) for p in parts if p.strip()]
            if nums:
                return np.mean(nums)
        except ValueError:
            return np.nan

    # Пробуем преобразовать в число
    try:
        return float(value_str)
    except ValueError:
        return np.nan


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Полная очистка данных:
    - Переименование колонок
    - Удаление дубликатов
    - Очистка числовых значений
    - Обработка пропусков
    - Удаление неинформативных колонок
    """
    df_clean = df.copy()

    # 1. Переименовываем колонки для удобства
    column_mapping = {
        'Company Names': 'Company',
        'Cars Names': 'Car_Name',
        'Engines': 'Engine',
        'CC/Battery Capacity': 'CC_Battery',
        'HorsePower': 'HP',
        'Total Speed': 'Speed',
        'Performance(0 - 100 )KM/H': 'Acceleration_0_100',
        'Cars Prices': 'Price',
        'Fuel Types': 'Fuel',
        'Seats': 'Seats',
        'Torque': 'Torque'
    }
    df_clean = df_clean.rename(columns=column_mapping)

    # 2. Удаляем дубликаты
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"Удалено дубликатов: {initial_rows - len(df_clean)}")

    # 3. Очищаем числовые колонки
    numeric_columns = ['CC_Battery', 'HP', 'Speed', 'Acceleration_0_100', 'Price', 'Torque', 'Seats']

    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(clean_numeric_value)

    # 4. Обрабатываем пропуски
    missing = df_clean.isnull().sum()
    if missing.sum() > 0:
        print(f"Пропуски в колонках:\n{missing[missing > 0]}")
        for col in df_clean.columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                mode_val = df_clean[col].mode()
                if not mode_val.empty:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
                else:
                    df_clean[col] = df_clean[col].fillna('Unknown')
    else:
        print("Пропусков после очистки нет")

    # 5. Удаляем неинформативные колонки
    cols_to_drop = ['Car_Name']  # Уникальные названия моделей
    df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns])

    # 6. Приводим Seats к целому
    if 'Seats' in df_clean.columns:
        df_clean['Seats'] = df_clean['Seats'].astype(int)

    print(f"Итоговый размер: {df_clean.shape[0]} строк, {df_clean.shape[1]} столбцов")
    return df_clean


def encode_categorical(df: pd.DataFrame, target_col: str = 'Price') -> pd.DataFrame:
    """
    Кодирование категориальных признаков.
    """
    df_encoded = df.copy()

    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    # Бинарные колонки через LabelEncoder
    binary_cols = []
    for col in categorical_cols:
        if df_encoded[col].nunique() == 2:
            binary_cols.append(col)

    for col in binary_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        categorical_cols.remove(col)

    # Остальные через One-Hot
    if categorical_cols:
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)

    return df_encoded


def split_data(df: pd.DataFrame, target_col: str = 'Price',
               test_size: float = 0.2, val_size: float = 0.2,
               random_state: int = 42):
    """Разделение на train/val/test."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=random_state
    )

    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio, random_state=random_state
    )

    print(f"Train: {len(X_train)} samples")
    print(f"Val:   {len(X_val)} samples")
    print(f"Test:  {len(X_test)} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """Стандартизация признаков."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler