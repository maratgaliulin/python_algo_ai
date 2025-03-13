import pandas as pd

def calculate_adx(data, period=14):
    """
    Рассчитывает ADX, +DI и -DI для DataFrame с ценами.
    
    :param data: DataFrame с колонками 'High', 'Low', 'Close'.
    :param period: Период для расчета ADX (по умолчанию 14).
    :return: DataFrame с колонками 'ADX', '+DI', '-DI'.


    Пояснение:
    
        True Range (TR) — максимальное значение из:
            Разницы между High и Low.
            Разницы между High и предыдущим Close.
            Разницы между Low и предыдущим Close.

        +DM и -DM — положительное и отрицательное направленное движение:
            +DM = High - High_prev, если это больше, чем Low_prev - Low.
            -DM = Low_prev - Low, если это больше, чем High - High_prev.

        +DI и -DI — индикаторы направления:
        +DI = (+DM_smooth / TR_smooth) * 100.
        -DI = (-DM_smooth / TR_smooth) * 100.

        DX — индекс направленного движения:
            DX = (|+DI - -DI| / (+DI + -DI)) * 100.

        ADX — сглаженное значение DX за указанный период.
    """
    # Вычисляем True Range (TR), +DM и -DM
    data['High-Low'] = data['High'] - data['Low']
    data['High-PrevClose'] = abs(data['High'] - data['Close'].shift(1))
    data['Low-PrevClose'] = abs(data['Low'] - data['Close'].shift(1))
    data['TR'] = data[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    
    data['+DM'] = data['High'] - data['High'].shift(1)
    data['-DM'] = data['Low'].shift(1) - data['Low']
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
    data.drop(['High-Low', 'High-PrevClose', 'Low-PrevClose', 'TR', '+DM', '-DM', 
               'TR_smooth', '+DM_smooth', '-DM_smooth', 'DX'], axis=1, inplace=True)
    
    return data

# Пример использования
if __name__ == "__main__":
    # Загрузка данных (пример)
    data = pd.DataFrame({
        'High': [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        'Low': [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        'Close': [29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5, 36.5, 37.5, 38.5, 39.5]
    })
    
    # Расчет ADX
    data_with_adx = calculate_adx(data, period=14)
    print(data_with_adx[['ADX', '+DI', '-DI']])