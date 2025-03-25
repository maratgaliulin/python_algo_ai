import MetaTrader5 as mt
import pandas as pd

def determine_the_deal(
    symb: str, 
    entry_price: float, 
    stoploss: float, 
    takeprofit: float, 
    order_type, 
    action, 
    magic: int, 
    lot: float,
    point: float,
    comment: str
) -> None:
    """
    Функция для определения и отправки сделки на основе переданных параметров.

    Входные аргументы:
        symb (str): Символ торгового инструмента.
        entry_price (float): Цена входа.
        stoploss (float): Цена стоп-лосса.
        takeprofit (float): Цена тейк-профита.
        last_ob_series (pd.Series): Серия данных последнего ордерблока.
        csv_address (str): Путь к CSV-файлу для записи данных.
        order_type: Тип ордера (например, на покупку или продажу).
        action: Действие с ордером (например, открытие или закрытие).
        magic (int): Уникальный идентификатор (магическое число) для ордера.
        lot (float): Размер лота.
        point (float): Размер пункта (минимальное изменение цены).
        comment (str): Комментарий к ордеру.

    Возвращает:
        None
    """

    # Получение текущих позиций и ордеров для указанного символа
    positions_of_the_symbol = mt.positions_get(symbol=symb)
    orders_of_the_symbol = mt.orders_get(symbol=symb)    

    deviation = 3  # Отклонение для ордера (в пунктах)
    # print('comment', comment)  # Вывод комментария для отладки

    # Формирование запроса на отправку ордера в зависимости от типа сделки (покупка или продажа)
    if comment == 'sell':
        # Запрос для ордера на продажу
        request = {
            "action": action,  # Действие (например, открытие ордера)
            "symbol": symb,  # Символ инструмента
            "volume": lot,  # Размер лота
            "type": order_type,  # Тип ордера (например, на продажу)
            "price": round(entry_price, 5),  # Цена входа, округленная до 5 знаков
            "price_stoplimit": round(entry_price, 5) - point,  # Цена для стоп-лимита (цена входа минус пункт)
            "sl": round(stoploss, 5),  # Цена стоп-лосса, округленная до 5 знаков
            "tp": round(takeprofit, 5),  # Цена тейк-профита, округленная до 5 знаков
            "deviation": deviation,  # Отклонение для ордера
            "comment": f"{comment}-{magic}",  # Комментарий к ордеру (тип сделки + магическое число)
            "type_time": mt.ORDER_TIME_GTC,  # Тип времени ордера (действителен до отмены)
            "type_filling": mt.ORDER_FILLING_IOC,  # Тип исполнения ордера (немедленно или отмена)
        } 
    else:
        # Запрос для ордера на покупку
        request = {
            "action": action,  # Действие (например, открытие ордера)
            "symbol": symb,  # Символ инструмента
            "volume": lot,  # Размер лота
            "type": order_type,  # Тип ордера (например, на покупку)
            "price": round(entry_price, 5),  # Цена входа, округленная до 5 знаков
            "price_stoplimit": round(entry_price, 5) + point,  # Цена для стоп-лимита (цена входа плюс пункт)
            "sl": round(stoploss, 5),  # Цена стоп-лосса, округленная до 5 знаков
            "tp": round(takeprofit, 5),  # Цена тейк-профита, округленная до 5 знаков
            "deviation": deviation,  # Отклонение для ордера
            "comment": f"{comment}-{magic}",  # Комментарий к ордеру (тип сделки + магическое число)
            "type_time": mt.ORDER_TIME_GTC,  # Тип времени ордера (действителен до отмены)
            "type_filling": mt.ORDER_FILLING_IOC,  # Тип исполнения ордера (немедленно или отмена)
        } 
    
    # Отправка запроса на открытие ордера
    result = mt.order_send(request)
    print('request sent', result)  # Вывод результата отправки запроса
    print(request)  # Вывод содержимого запроса для отладки