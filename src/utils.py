import random
import numpy as np

def set_seed(seed: int = 42):
    """
    Фиксирует seed для воспроизводимости результатов.
    """
    random.seed(seed)
    np.random.seed(seed)
    
def calculate_metrics(y_true, y_pred):
    """
    Возвращает словарь с метриками регрессии.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    # Исправление: считаем RMSE вручную через np.sqrt
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': round(mae, 2),
        'RMSE': round(float(rmse), 2),  # <-- добавили float()
        'R2': round(r2, 4)
    }