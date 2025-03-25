import pandas as pd
import numpy as np
from methods.order_placement_buy import order_placement_buy
from methods.order_placement_sell import order_placement_sell
from methods.determine_the_deal import determine_the_deal

def buy_or_sell(
    min_impulse_size: float,
    orders_of_the_symbol:tuple,
    positions_of_the_symbol: tuple,
    present_price_bid:float,
    present_price_ask:float,
    point:float,
    order_type_buy,
    order_type_sell,
    order_action,
    lot:float,
    magic:int,
    high_value:float,
    low_value:float,
    trend_direction:np.ndarray
                ) -> None:
    
    entry, sl, tp = 0, 0, 0    
    
    trend_direction = trend_direction.tolist()
    actual_impulse_size = high_value - low_value
    
    if(trend_direction[0] == [1, 0, 0]):
        trend_direction_string = 'downtrend'
    elif(trend_direction[0] == [0, 1, 0]):
        trend_direction_string = 'uptrend'
    elif(trend_direction[0] == [0, 0, 1]):
        trend_direction_string = 'undefined'
    else:
        trend_direction_string = 'error'
    
    impulse_satisfies_minimal_size = actual_impulse_size >= min_impulse_size
    orders_of_the_symbol_is_null = orders_of_the_symbol == ()
    positions_of_the_symbol_is_null = positions_of_the_symbol == ()
    trend_is_predicted = trend_direction_string == 'uptrend' or trend_direction_string == 'downtrend'
    
    print(f'Predicted high value: {high_value}.')
    print(f'Predicted low value: {low_value}.')    
    print(f'Impulse satisfies minimal_size: {impulse_satisfies_minimal_size}.')
    print(f'Impulse minimal size: {min_impulse_size}, predicted impulse size: {actual_impulse_size}.')
    print(f'There are no open orders: {orders_of_the_symbol_is_null}.')
    print(f'There are no open positions: {positions_of_the_symbol_is_null}.')
    print(f'Trend is predicted: {trend_is_predicted}. Actual trend direction: {trend_direction_string}, {trend_direction}.')
    
    conditions_for_order_placement = impulse_satisfies_minimal_size and orders_of_the_symbol_is_null and positions_of_the_symbol_is_null and trend_is_predicted
    
    print(f'conditions_for_order_placement: {conditions_for_order_placement}.')
    
    if((conditions_for_order_placement == True) or (conditions_for_order_placement == [True])):
        if(trend_direction_string == 'uptrend'):
            
            entry, sl, tp = order_placement_buy(price_impulse_start=low_value, price_impulse_end=high_value, point=point)
            
            if(present_price_ask >= entry):
            
                determine_the_deal(
                    symbol="EURUSD", 
                    entry_price=entry,
                    stoploss=sl,
                    takeprofit=tp,
                    order_type=order_type_buy,
                    action=order_action,
                    magic=magic,
                    lot=lot,
                    point=point,
                    comment='buy'
                )
                
        elif(trend_direction_string == 'downtrend'):
            
            entry, sl, tp = order_placement_sell(price_impulse_start=high_value, price_impulse_end=low_value, point=point)
            
            if(present_price_bid <= entry):
            
                determine_the_deal(
                    symbol="EURUSD", 
                    entry_price=entry,
                    stoploss=sl,
                    takeprofit=tp,
                    order_type=order_type_sell,
                    action=order_action,
                    magic=magic,
                    lot=lot,
                    point=point,
                    comment='sell'
                )
    
    