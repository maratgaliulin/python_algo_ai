import pandas as pd
import numpy as np
from methods.order_placement_buy import order_placement_buy
from methods.order_placement_sell import order_placement_sell
from methods.determine_the_deal import determine_the_deal
from methods.modify_stoploss import modify_stoploss

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
    high_price:float,
    low_price:float,
    trend_direction:str
                ) -> int:
    
    entry, sl, tp = 0, 0, 0    

    time_sleep = 0
    
    # trend_direction = trend_direction.tolist()
    high_price_pres_price_difference = abs(high_price - present_price_ask)
    low_price_pres_price_difference = abs(low_price - present_price_bid)

    if (high_price_pres_price_difference > low_price_pres_price_difference):
        actual_impulse_size = high_price_pres_price_difference
    else:
        actual_impulse_size = low_price_pres_price_difference
    
    # if(trend_direction[0] == [1, 0, 0]):
    #     trend_direction_string = 'downtrend'
    # elif(trend_direction[0] == [0, 1, 0]):
    #     trend_direction_string = 'uptrend'
    # elif(trend_direction[0] == [0, 0, 1]):
    #     trend_direction_string = 'undefined'
    # else:
    #     trend_direction_string = 'error'
    

    impulse_satisfies_minimal_size = actual_impulse_size >= min_impulse_size
    orders_of_the_symbol_is_null = orders_of_the_symbol == ()
    positions_of_the_symbol_is_null = positions_of_the_symbol == ()
    middle_price = (high_price - low_price) / 2 + low_price
    # trend_is_predicted = trend_direction_string == 'uptrend' or trend_direction_string == 'downtrend'
    # trend_is_predicted = trend_direction[0] == 'uptrend' or trend_direction[0] == 'downtrend'
    trend_is_predicted = trend_direction == 'uptrend' or trend_direction == 'downtrend'
    present_price_top_difference = high_price - present_price_ask
    present_price_bottom_difference = present_price_bid - low_price
    present_price_bid_to_middle_value_difference = abs(middle_price - present_price_bid)
    present_price_ask_to_middle_value_difference = abs(middle_price - present_price_ask)


    present_price_is_closer_to_top = present_price_top_difference < present_price_bottom_difference
    present_price_is_closer_to_bottom = present_price_top_difference > present_price_bottom_difference
    present_price_is_closer_to_middle_price = (
        (present_price_bid_to_middle_value_difference < abs(present_price_top_difference)) and
        (present_price_ask_to_middle_value_difference < abs(present_price_top_difference)) and 
        (present_price_bid_to_middle_value_difference < abs(present_price_bottom_difference)) and
        (present_price_ask_to_middle_value_difference < abs(present_price_bottom_difference)))

    # print(f'Predicted high value: {high_value}.')
    # print(f'Predicted low value: {low_value}.')   
    print('***********************************') 
    print(f'Impulse satisfies minimal_size: {impulse_satisfies_minimal_size}.')
    print(f'Impulse minimal size: {min_impulse_size}, predicted impulse size: {actual_impulse_size}.')
    print('***********************************') 
    print(f'There are no open orders: {orders_of_the_symbol_is_null}.')
    print(f'There are no open positions: {positions_of_the_symbol_is_null}.')
    print('***********************************') 
    # print(f'Trend is predicted: {trend_is_predicted}. Actual trend direction: {trend_direction_string}, {trend_direction}.')
    print(f'Trend is predicted: {trend_is_predicted}. Actual trend direction: {trend_direction}.')
    print('***********************************') 
    print(f'Present price is closer to predicted high value: {present_price_is_closer_to_top}')
    print(f'Present price is closer to predicted low value: {present_price_is_closer_to_bottom}')
    print(f'Present price is closer to middle price: {present_price_is_closer_to_middle_price}')
    # print('***********************************') 

    conditions_for_order_placement = (
                                        impulse_satisfies_minimal_size and 
                                      orders_of_the_symbol_is_null and 
                                      positions_of_the_symbol_is_null)
    
    print(f'conditions_for_order_placement: {conditions_for_order_placement}.')
    
    if(conditions_for_order_placement):
        if(present_price_is_closer_to_bottom):
        # if(trend_direction == 'uptrend'):

            entry, sl, tp = order_placement_buy(price_impulse_start=present_price_ask, price_impulse_end=high_price, point=point)

            print(f'Trend direction: {trend_direction}. Entry price: {entry}. Stoploss: {sl}. Takeprofit: {tp}. Impulse size: {abs(entry - tp)}.')

            determine_the_deal(
                symb="EURUSD", 
                entry_price=entry,
                stoploss=sl,
                takeprofit=float(tp),
                order_type=order_type_buy,
                action=order_action,
                magic=magic,
                lot=lot,
                point=point,
                comment='buy'
            )

               
        elif(present_price_is_closer_to_top):
        # elif(trend_direction == 'downtrend'):
            
            entry, sl, tp = order_placement_sell(price_impulse_start=present_price_bid, price_impulse_end=low_price, point=point)

            print(f'Trend direction: {trend_direction}. Entry price: {entry}. Stoploss: {sl}. Takeprofit: {tp}. Impulse size: {abs(entry - tp)}.')

            # print('Present price is closer to predicted high value. Checking if it is less than or equal to entry price.')
            print(f'Entry price: {entry}')
            print(f'Present price bid: {present_price_bid}')
            
            determine_the_deal(
                symb="EURUSD", 
                entry_price=entry,
                stoploss=sl,
                takeprofit=float(tp),
                order_type=order_type_sell,
                action=order_action,
                magic=magic,
                lot=lot,
                point=point,
                comment='sell'
            )

               
    if(positions_of_the_symbol != ()):
        positions_of_the_symbol_dict = positions_of_the_symbol[0]._asdict()                       
        entry = positions_of_the_symbol_dict['price_open']            
        stoploss = positions_of_the_symbol_dict['sl']            
        takeprofit = positions_of_the_symbol_dict['tp']

        position_comment = positions_of_the_symbol_dict['comment'].split('-')[0]

        print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-')

        print('position comment:', position_comment)

        print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-')            
        
        if(position_comment == 'buy'):  
            sl_tp_diff = takeprofit - entry
            
            sl_pres_price_ask_diff = present_price_ask - stoploss
            
            if(sl_pres_price_ask_diff > 150 * point):
                modified_stoploss = stoploss + 50 * point
                modified_takeprofit = takeprofit
            
                modify_stoploss(positions_of_the_symbol_dict, modified_stoploss, modified_takeprofit)   
                print(f'Сработало условие - стоплосс на покупку сдвинут на линию 50 пунктов вверх, а тейк равен {takeprofit}')
                
        elif(position_comment == 'sell'):     
            
            sl_pres_price_bid_diff = stoploss - present_price_bid
            if(sl_pres_price_bid_diff > 150 * point):
                modified_stoploss = stoploss - 50 * point
                modified_takeprofit = takeprofit
            
                modify_stoploss(positions_of_the_symbol_dict, modified_stoploss, modified_takeprofit)   
                print(f'Сработало условие - стоплосс на продажу сдвинут на 50 пунктов вниз, а тейк равен {takeprofit}')
    
    return time_sleep
    
    