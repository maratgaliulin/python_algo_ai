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
    high_value:float,
    low_value:float,
    trend_direction:np.ndarray
                ) -> None:
    
    entry, sl, tp = 0, 0, 0    
    
    trend_direction = trend_direction.tolist()
    actual_impulse_size = high_value - low_value
    
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
    # trend_is_predicted = trend_direction_string == 'uptrend' or trend_direction_string == 'downtrend'
    trend_is_predicted = trend_direction[0] == 'uptrend' or trend_direction[0] == 'downtrend'
    present_price_top_difference = high_value - present_price_ask
    present_price_bottom_difference = present_price_bid - low_value
    present_price_is_closer_to_top = present_price_top_difference < present_price_bottom_difference
    present_price_is_closer_to_bottom = present_price_top_difference > present_price_bottom_difference
    
    print(f'Predicted high value: {high_value}.')
    print(f'Predicted low value: {low_value}.')   
    print('***********************************') 
    print(f'Impulse satisfies minimal_size: {impulse_satisfies_minimal_size}.')
    print(f'Impulse minimal size: {min_impulse_size}, predicted impulse size: {actual_impulse_size}.')
    print('***********************************') 
    print(f'There are no open orders: {orders_of_the_symbol_is_null}.')
    print(f'There are no open positions: {positions_of_the_symbol_is_null}.')
    print('***********************************') 
    # print(f'Trend is predicted: {trend_is_predicted}. Actual trend direction: {trend_direction_string}, {trend_direction}.')
    print(f'Trend is predicted: {trend_is_predicted}. Actual trend direction: {trend_direction[0]}.')
    print('***********************************') 
    print(f'Present price is closer to predicted high value: {present_price_is_closer_to_top}')
    print(f'Present price is closer to predicted low value: {present_price_is_closer_to_bottom}')
    print('***********************************') 

    conditions_for_order_placement = impulse_satisfies_minimal_size and orders_of_the_symbol_is_null and positions_of_the_symbol_is_null
    
    print(f'conditions_for_order_placement: {conditions_for_order_placement}.')
    
    if(conditions_for_order_placement):
        if(present_price_is_closer_to_bottom):
            
            entry, sl, tp = order_placement_buy(price_impulse_start=low_value, price_impulse_end=high_value, point=point)
            
            # print('Present price is closer to predicted low value. Checking if it is greater than or equal to entry price.')
            print(f'Entry price: {entry}')
            print(f'Present price ask: {present_price_ask}')

            if(present_price_ask >= entry):
                print('Present price is greater than or equal to entry price. Placing the order.')

                determine_the_deal(
                    symb="EURUSD", 
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

            else:
                print('Present price is less than entry price. Placing the order.')

                determine_the_deal(
                    symb="EURUSD", 
                    entry_price=present_price_ask,
                    stoploss=present_price_ask - 50 * point,
                    takeprofit=tp,
                    order_type=order_type_buy,
                    action=order_action,
                    magic=magic,
                    lot=lot,
                    point=point,
                    comment='buy'
                )
                
        elif(present_price_is_closer_to_top):
            
            entry, sl, tp = order_placement_sell(price_impulse_start=high_value, price_impulse_end=low_value, point=point)

            # print('Present price is closer to predicted high value. Checking if it is less than or equal to entry price.')
            print(f'Entry price: {entry}')
            print(f'Present price bid: {present_price_bid}')
            
            if(present_price_bid <= entry):

                print('Present price is less than or equal to entry price. Placing the order.')
            
                determine_the_deal(
                    symb="EURUSD", 
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

            else:

                print('Present price is greater than entry price. Placing the order.')
            
                determine_the_deal(
                    symb="EURUSD", 
                    entry_price=present_price_bid,
                    stoploss=present_price_bid + 50 * point,
                    takeprofit=tp,
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
            modified_stoploss = stoploss
            modified_takeprofit = takeprofit
            line_above_stoploss = entry - 10 * point
            line_0_5 = entry + point   # line_entry
            line_0_618 = entry + sl_tp_diff * 0.146
            line_0_764 = entry + sl_tp_diff * 0.236 
            line_0_854 = entry + sl_tp_diff * 0.382
            line_0_932 = entry + sl_tp_diff * 0.5
            line_1_0 = entry + sl_tp_diff * 0.618
            line_1_146 = entry + sl_tp_diff * 0.764
            line_1_236 = entry + sl_tp_diff * 0.854
            line_1_382 = entry + sl_tp_diff * 0.932
            line_1_50 = entry + sl_tp_diff
            line_1_6 = entry + sl_tp_diff * 1.146
            line_1_618 = takeprofit + sl_tp_diff * 1.236
            line_1_764 = takeprofit + sl_tp_diff * 1.382

            print('line_1_0' , line_1_0)

            if((present_price_ask >= line_0_854) and (present_price_ask < line_0_932)):
                if(modified_stoploss < line_above_stoploss):
                    modified_stoploss = line_0_5                  

            elif((present_price_ask >= line_0_932) and (present_price_ask < line_1_0)):
                if((modified_stoploss <= line_0_5 + 5*point) and (modified_stoploss > line_above_stoploss)): 
                    modified_stoploss = line_0_618                  

            elif((present_price_ask >= line_1_0) and (present_price_ask < line_1_146)): 
                if((modified_stoploss <= line_0_618 + 5*point) and (modified_stoploss > line_0_5)): 
                    modified_stoploss = line_0_764
                    
            elif((present_price_ask >= line_1_146) and (present_price_ask < line_1_236)):
                if((modified_stoploss <= line_0_764 + 5*point) and (modified_stoploss > line_0_618)): 
                    modified_stoploss = line_0_854

            elif((present_price_ask >= line_1_236) and (present_price_ask < line_1_382)):
                if((modified_stoploss <= line_0_854 + 5*point) and (modified_stoploss > line_0_764)):
                    modified_stoploss = line_1_0    

            elif((present_price_ask >= line_1_382) and (present_price_ask < line_1_50)):
                if((modified_stoploss <= line_1_0 + 5*point) and (modified_stoploss > line_0_854)):
                    modified_stoploss = line_1_146
                
            print(stoploss)
            print(modified_stoploss)
            if(modified_stoploss != stoploss):
                fibo_level = round((modified_stoploss - stoploss)/sl_tp_diff, 3)
                modify_stoploss(positions_of_the_symbol_dict, modified_stoploss, modified_takeprofit)   
                print(f'Сработало условие - стоплосс на покупку сдвинут на линию {fibo_level} по фибе, а тейк равен {takeprofit}')
                
        elif(position_comment == 'sell'):     
            sl_tp_diff = entry - takeprofit
            modified_stoploss = stoploss
            modified_takeprofit = takeprofit
            line_below_stoploss = entry + 10 * point
            line_0_5 = entry - point   # line_entry
            line_0_618 = entry - sl_tp_diff * 0.146  # line_0
            line_0_764 = entry - sl_tp_diff * 0.236   # line_0_618
            line_0_854 = entry - sl_tp_diff * 0.382
            line_0_932 = entry - sl_tp_diff * 0.5
            line_1_0 = entry - sl_tp_diff * 0.618
            line_1_146 = entry - sl_tp_diff * 0.764
            line_1_236 = entry - sl_tp_diff * 0.854
            line_1_382 = entry - sl_tp_diff * 0.932
            line_1_50 = entry - sl_tp_diff
            line_1_6 = entry - sl_tp_diff * 1.146
            line_1_618 = takeprofit - sl_tp_diff * 1.236
            line_1_764 = takeprofit - sl_tp_diff * 1.382

            

            if ((present_price_bid <= line_0_854) and (present_price_bid > line_0_932)): 
                if(modified_stoploss > line_below_stoploss):  
                    modified_stoploss = line_0_5                    
                
            elif((present_price_bid <= line_0_932) and (present_price_bid > line_1_0)): 
                if((modified_stoploss >= line_0_5 - 5*point) and (modified_stoploss < line_below_stoploss)):
                    modified_stoploss = line_0_618       

            elif((present_price_bid <= line_1_0) and (present_price_bid > line_1_146)):
                if((modified_stoploss >= line_0_618 - 5*point) and (modified_stoploss < line_0_5)):
                    modified_stoploss = line_0_764                
            
            elif((present_price_bid <= line_1_146) and (present_price_bid > line_1_236)):
                if((modified_stoploss >= line_0_764 - 5*point) and (modified_stoploss < line_0_618)):
                    modified_stoploss = line_0_854

            elif((present_price_bid <= line_1_236) and (present_price_bid > line_1_382)):
                if((modified_stoploss >= line_0_854 - 5*point) and (modified_stoploss < line_0_764)):
                    modified_stoploss = line_1_0    

            elif((present_price_bid <= line_1_382) and (present_price_bid > line_1_50)):
                if((modified_stoploss >= line_1_0 - 5*point) and (modified_stoploss < line_0_854)):
                    modified_stoploss = line_1_146

            fibo_level = (stoploss - modified_stoploss)/sl_tp_diff
            print('fibo level:', fibo_level)

            if(modified_stoploss != stoploss):
                fibo_level = round((stoploss - modified_stoploss)/sl_tp_diff, 3)
                modify_stoploss(positions_of_the_symbol_dict, modified_stoploss, modified_takeprofit)
                print(f'Сработало условие - стоплосс на продажу сдвинут на линию {fibo_level} по фибе, а тейк равен {takeprofit}')
    
    