import pandas as pd
import MetaTrader5 as mt
# from .make_5min_volume_from_1_min_volume import resample_time_series
from .make_clean_dataframe_from_server import make_clean_dataframe_from_server
from .build_dynamic_profile import build_dynamic_profile
import numpy as np
import ta


def return_entry_point_and_takeprofit(predicted_high_price:float, predicted_low_price:float, present_price_bid:float, present_price_ask:float, start_pos:int, end_pos:int) -> tuple[float, float, str]:

    SYMBOL = 'EURUSD'
    TIMEFRAME_LONG_MT = mt.TIMEFRAME_M15
    num_bins_total = 30
    
    df_joined_eurusd_15min = make_clean_dataframe_from_server(symbol=SYMBOL, timeframe=TIMEFRAME_LONG_MT, start_pos=start_pos, end_pos=end_pos)

    n_bins, _ = build_dynamic_profile(data_slice=df_joined_eurusd_15min, num_bins=num_bins_total)
    
    abs_bid_present_price_difference = abs(predicted_low_price - present_price_bid)

    abs_ask_present_price_difference = abs(predicted_high_price - present_price_ask)




    abs_predicted_high_price_difference = abs(predicted_high_price - n_bins[0])

    abs_predicted_low_price_difference = abs(predicted_low_price - n_bins[0])

    abs_predicted_high_price_difference_index = 0
    abs_predicted_low_price_difference_index = 0

    trend_direction = 'undefined'

    if(abs_bid_present_price_difference > abs_ask_present_price_difference):
        # Если предсказанная цена наверху, то покупаем, поэтому действия осуществляем с ценой ask

        minimal_price_difference = abs(n_bins[0] - present_price_ask)

        minimal_price_difference_index = 0
        trend_direction = 'uptrend'

        # вычисляем зону ликвидности, ближайшую к текущей цене ask:
        for idx in range(len(n_bins)):
            current_price_difference = abs(n_bins[idx] - present_price_ask)   

            if (current_price_difference < minimal_price_difference):

                minimal_price_difference = current_price_difference
                minimal_price_difference_index = idx

        # вычисляем зону ликвидности, ближайшую к предсказанной цене high:
        for idx in range(len(n_bins)):
            current_predicted_price_difference = abs(predicted_high_price - n_bins[idx])
            if (current_predicted_price_difference < abs_predicted_high_price_difference):
                abs_predicted_high_price_difference = current_predicted_price_difference
                abs_predicted_high_price_difference_index = idx
        
        entry_point = n_bins[minimal_price_difference_index]
        take_profit = n_bins[abs_predicted_high_price_difference_index]

        return entry_point, take_profit, trend_direction
                

    else:
        minimal_price_difference = abs(n_bins[0] - present_price_bid)
        minimal_price_difference_index = 0
        trend_direction = 'downtrend'

        for idx in range(len(n_bins)):
            current_price_difference = abs(n_bins[idx] - present_price_bid)
            if (current_price_difference < minimal_price_difference):
                minimal_price_difference = current_price_difference
                minimal_price_difference_index = idx
            # print('current_price_difference:', current_price_difference)
            # print('minimal_price_difference_index:', minimal_price_difference_index)

        # вычисляем зону ликвидности, ближайшую к предсказанной цене low:
        for idx in range(len(n_bins)):
            current_predicted_price_difference = abs(predicted_low_price - n_bins[idx])
            if (current_predicted_price_difference < abs_predicted_low_price_difference):
                abs_predicted_low_price_difference = current_predicted_price_difference
                abs_predicted_low_price_difference_index = idx
                

        entry_point = n_bins[minimal_price_difference_index]
        take_profit = n_bins[abs_predicted_low_price_difference_index]
    
    
        return entry_point, take_profit, trend_direction