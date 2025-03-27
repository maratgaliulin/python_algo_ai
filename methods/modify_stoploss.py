import MetaTrader5 as mt

def modify_stoploss(positions_of_the_symbol, stoploss: float, takeprofit: float) -> None:
    """
    Функция для модификации уровня стоп-лосса и тейк-профита у открытой позиции.

    Параметры:
    - positions_of_the_symbol (dict): Словарь с информацией о позиции (символ, тикет и т.д.).
    - stoploss (float): Новый уровень стоп-лосса.
    - takeprofit (float): Новый уровень тейк-профита.

    Возвращает:
    - None: Функция не возвращает значений, но выводит результат выполнения запроса в консоль.
    """

    # Вывод сообщения о начале модификации стоп-лосса
    print('stoploss is being modified')

    # Получение символа из информации о позиции
    symb = positions_of_the_symbol['symbol']

    # Формирование запроса на модификацию стоп-лосса и тейк-профита
    request = {
        "action": mt.TRADE_ACTION_SLTP,  # Тип действия: модификация SL/TP
        'symbol': symb,  # Символ инструмента
        "ticket": positions_of_the_symbol['ticket'],  # Тикет позиции
        "position": positions_of_the_symbol['ticket'],  # Идентификатор позиции
        "order": positions_of_the_symbol['ticket'],  # Идентификатор ордера
        "sl": round(stoploss, 5),  # Новый уровень стоп-лосса (округленный до 5 знаков)
        'tp': takeprofit,  # Новый уровень тейк-профита
        "comment": 'stoploss was reset'  # Комментарий к операции
    }

    # Отправка запроса на модификацию
    result = mt.order_send(request)

    # Вывод результата выполнения запроса в консоль
    print(result._asdict())
    