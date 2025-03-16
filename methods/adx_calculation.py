import pandas as pd

def calculate_adx(data: pd.DataFrame, period=14) -> pd.DataFrame:
    """
    Рассчитывает ADX, +DI и -DI для DataFrame с ценами.
    
    :param data: DataFrame с колонками 'high', 'low', 'close'.
    :param period: Период для расчета ADX (по умолчанию 14).
    :return: DataFrame с колонками 'ADX', '+DI', '-DI'.


    Пояснение:
    
        True Range (TR) — максимальное значение из:
            Разницы между high и low.
            Разницы между high и предыдущим close.
            Разницы между low и предыдущим close.

        +DM и -DM — положительное и отрицательное направленное движение:
            +DM = high - high_prev, если это больше, чем low_prev - low.
            -DM = low_prev - low, если это больше, чем high - high_prev.

        +DI и -DI — индикаторы направления:
        +DI = (+DM_smooth / TR_smooth) * 100.
        -DI = (-DM_smooth / TR_smooth) * 100.

        DX — индекс направленного движения:
            DX = (|+DI - -DI| / (+DI + -DI)) * 100.

        ADX — сглаженное значение DX за указанный период.
    """
    # Вычисляем True Range (TR), +DM и -DM
    data['high-low'] = data['high'] - data['low']
    data['high-Prevclose'] = abs(data['high'] - data['close'].shift(1))
    data['low-Prevclose'] = abs(data['low'] - data['close'].shift(1))
    data['TR'] = data[['high-low', 'high-Prevclose', 'low-Prevclose']].max(axis=1)
    
    data['+DM'] = data['high'] - data['high'].shift(1)
    data['-DM'] = data['low'].shift(1) - data['low']
    data['+DM'] = data['+DM'].where(data['+DM'] > data['-DM'], 0)
    data['-DM'] = data['-DM'].where(data['-DM'] > data['+DM'], 0)
    
    # Сглаживаем TR, +DM и -DM с использованием скользящих средних
    data['TR_smooth'] = data['TR'].rolling(window=period, min_periods=1).sum()
    data['+DM_smooth'] = data['+DM'].rolling(window=period, min_periods=1).sum()
    data['-DM_smooth'] = data['-DM'].rolling(window=period, min_periods=1).sum()
    
    # Рассчитываем +DI и -DI
    data['+DI'] = (data['+DM_smooth'] / data['TR_smooth']) * 100
    data['-DI'] = (data['-DM_smooth'] / data['TR_smooth']) * 100
    
    # Рассчитываем Directional Movement Index (DX)
    data['DX'] = (abs(data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI'])) * 100
    
    # Рассчитываем ADX как сглаженный DX
    data['ADX'] = data['DX'].rolling(window=period, min_periods=1).mean()
    
    # Убираем временные колонки
    data.drop(['high-low', 'high-Prevclose', 'low-Prevclose', 'TR', '+DM', '-DM', 
               'TR_smooth', '+DM_smooth', '-DM_smooth', 'DX'], axis=1, inplace=True)
    
    return data

# Пример использования
# if __name__ == "__main__":
#     # Загрузка данных (пример)
#     data = pd.DataFrame({
#         'high': [30, 31, 32, 33, 34, 33, 32, 31, 30, 29, 28],
#         'low': [29, 30, 31, 32, 33, 32, 31, 30, 29, 28, 27],
#         'close': [29.5, 30.5, 31.5, 32.5, 33.5, 32.5, 31.5, 30.5, 29, 28.5, 27.5]
#     })
    
#     # Расчет ADX
#     data_with_adx = calculate_adx(data, period=14)
#     print(data_with_adx)