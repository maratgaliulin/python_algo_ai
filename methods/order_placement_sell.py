def order_placement_sell(price_impulse_start: float, price_impulse_end: float, point: float):
    """
    Функция для расчета уровней входа, стоп-лосса и тейк-профита на основе уровней Фибоначчи.

    Параметры:
    - price_impulse_start (float): Цена начала импульса (начало движения).
    - price_impulse_end (float): Цена окончания импульса (конец движения).
    - point (float): Размер одного пункта (используется для точности расчетов, но в текущей реализации не используется).

    Возвращает:
    - entry_point (float): Цена для входа в сделку (уровень 50% от коррекции Фибоначчи).
    - stop_loss (float): Уровень стоп-лосса (цена начала импульса).
    - take_profit (float): Уровень тейк-профита (уровень 161.8% от импульса).
    """

    # Расчет точки входа
    entry_point = price_impulse_start - 5 * point

    # Уровень стоп-лосса устанавливается на 20 пипсов ниже начала импульса
    stop_loss = price_impulse_start + 20 * point

    # Расчет уровня тейк-профита
    take_profit = price_impulse_end + 5 * point

    # Возвращаем рассчитанные уровни
    return entry_point, stop_loss, take_profit