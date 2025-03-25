import pandas as pd

def define_the_trend(df:pd.DataFrame):

    high_low_0 = (df['high'] + df['low'])/2
    high_low_5 = (df['high_plus_5min'] + df['low_plus_5min'])/2
    high_low_10 = (df['high_plus_10min'] + df['low_plus_10min'])/2
    high_low_15 = (df['high_plus_15min'] + df['low_plus_15min'])/2
    high_low_20 = (df['high_plus_20min'] + df['low_plus_20min'])/2
    high_low_25 = (df['high_plus_25min'] + df['low_plus_25min'])/2    
    high_low_30 = (df['high_plus_30min'] + df['low_plus_30min'])/2
    high_low_35 = (df['high_plus_35min'] + df['low_plus_35min'])/2
    high_low_40 = (df['high_plus_40min'] + df['low_plus_40min'])/2    
    high_low_45 = (df['high_plus_45min'] + df['low_plus_45min'])/2
    high_low_50 = (df['high_plus_50min'] + df['low_plus_50min'])/2
    high_low_55 = (df['high_plus_55min'] + df['low_plus_55min'])/2    
    high_low_60 = (df['high_plus_60min'] + df['low_plus_60min'])/2
    
    last_4_candles_increase = (
        (high_low_45 < high_low_50) and (high_low_50 < high_low_55) and (high_low_55 < high_low_60)
    )
    
    last_5_candles_increase = (
        (high_low_40 < high_low_45) and (high_low_45 < high_low_50) and (high_low_50 < high_low_55) and (high_low_55 < high_low_60)
    )
    
    last_6_candles_increase = (
        (high_low_35 < high_low_40) and (high_low_40 < high_low_45) and (high_low_45 < high_low_50) and (high_low_50 < high_low_55) and (high_low_55 < high_low_60)
    )
    
    last_7_candles_increase = (
        (high_low_30 < high_low_35) and (high_low_35 < high_low_40) and (high_low_40 < high_low_45) and (high_low_45 < high_low_50) and (high_low_50 < high_low_55) and (high_low_55 < high_low_60)
    )
    
    last_8_candles_increase = (
        (high_low_25 < high_low_30) and (high_low_30 < high_low_35) and (high_low_35 < high_low_40) and (high_low_40 < high_low_45) and (high_low_45 < high_low_50) and (high_low_50 < high_low_55) and (high_low_55 < high_low_60)
    )
    
    last_9_candles_increase = (
        (high_low_20 < high_low_25) and (high_low_25 < high_low_30) and (high_low_30 < high_low_35) and (high_low_35 < high_low_40) and (high_low_40 < high_low_45) and (high_low_45 < high_low_50) and (high_low_50 < high_low_55) and (high_low_55 < high_low_60)
    )
    
    last_10_candles_increase = (
        (high_low_15 < high_low_20) and (high_low_20 < high_low_25) and (high_low_25 < high_low_30) and (high_low_30 < high_low_35) and (high_low_35 < high_low_40) and (high_low_40 < high_low_45) and (high_low_45 < high_low_50) and (high_low_50 < high_low_55) and (high_low_55 < high_low_60)
    )
    
    last_11_candles_increase = (
        (high_low_10 < high_low_15) and (high_low_15 < high_low_20) and (high_low_20 < high_low_25) and (high_low_25 < high_low_30) and (high_low_30 < high_low_35) and (high_low_35 < high_low_40) and (high_low_40 < high_low_45) and (high_low_45 < high_low_50) and (high_low_50 < high_low_55) and (high_low_55 < high_low_60)
    )
    
    last_12_candles_increase = (
        (high_low_5 < high_low_10) and (high_low_10 < high_low_15) and (high_low_15 < high_low_20) and (high_low_20 < high_low_25) and (high_low_25 < high_low_30) and (high_low_30 < high_low_35) and (high_low_35 < high_low_40) and (high_low_40 < high_low_45) and (high_low_45 < high_low_50) and (high_low_50 < high_low_55) and (high_low_55 < high_low_60)
    )
    
    
    last_4_candles_decrease = (
        (high_low_45 > high_low_50) and (high_low_50 > high_low_55) and (high_low_55 > high_low_60)
    )
    
    last_5_candles_decrease = (
        (high_low_40 > high_low_45) and (high_low_45 > high_low_50) and (high_low_50 > high_low_55) and (high_low_55 > high_low_60)
    )
    
    last_6_candles_decrease = (
        (high_low_35 > high_low_40) and (high_low_40 > high_low_45) and (high_low_45 > high_low_50) and (high_low_50 > high_low_55) and (high_low_55 > high_low_60)
    )
    
    last_7_candles_decrease = (
        (high_low_30 > high_low_35) and (high_low_35 > high_low_40) and (high_low_40 > high_low_45) and (high_low_45 > high_low_50) and (high_low_50 > high_low_55) and (high_low_55 > high_low_60)
    )
    
    last_8_candles_decrease = (
        (high_low_25 > high_low_30) and (high_low_30 > high_low_35) and (high_low_35 > high_low_40) and (high_low_40 > high_low_45) and (high_low_45 > high_low_50) and (high_low_50 > high_low_55) and (high_low_55 > high_low_60)
    )
    
    last_9_candles_decrease = (
        (high_low_20 > high_low_25) and (high_low_25 > high_low_30) and (high_low_30 > high_low_35) and (high_low_35 > high_low_40) and (high_low_40 > high_low_45) and (high_low_45 > high_low_50) and (high_low_50 > high_low_55) and (high_low_55 > high_low_60)
    )
    
    last_10_candles_decrease = (
        (high_low_15 > high_low_20) and (high_low_20 > high_low_25) and (high_low_25 > high_low_30) and (high_low_30 > high_low_35) and (high_low_35 > high_low_40) and (high_low_40 > high_low_45) and (high_low_45 > high_low_50) and (high_low_50 > high_low_55) and (high_low_55 > high_low_60)
    )
    
    last_11_candles_decrease = (
        (high_low_10 > high_low_15) and (high_low_15 > high_low_20) and (high_low_20 > high_low_25) and (high_low_25 > high_low_30) and (high_low_30 > high_low_35) and (high_low_35 > high_low_40) and (high_low_40 > high_low_45) and (high_low_45 > high_low_50) and (high_low_50 > high_low_55) and (high_low_55 > high_low_60)
    )
    
    last_12_candles_decrease = (
        (high_low_5 > high_low_10) and (high_low_10 > high_low_15) and (high_low_15 > high_low_20) and (high_low_20 > high_low_25) and (high_low_25 > high_low_30) and (high_low_30 > high_low_35) and (high_low_35 > high_low_40) and (high_low_40 > high_low_45) and (high_low_45 > high_low_50) and (high_low_50 > high_low_55) and (high_low_55 > high_low_60)
    )

    average_price_is_increasing = (
        last_4_candles_increase or
        last_5_candles_increase or
        last_6_candles_increase or
        last_7_candles_increase or
        last_8_candles_increase or
        last_9_candles_increase or
        last_10_candles_increase or
        last_11_candles_increase or
        last_12_candles_increase
    )
    
    average_price_is_decreasing = (
        last_4_candles_decrease or
        last_5_candles_decrease or
        last_6_candles_decrease or
        last_7_candles_decrease or
        last_8_candles_decrease or
        last_9_candles_decrease or
        last_10_candles_decrease or
        last_11_candles_decrease or
        last_12_candles_decrease
    )
    
    if(average_price_is_increasing):
        return 'uptrend'
    elif(average_price_is_decreasing):
        return 'downtrend'
    else:
        return 'trend undefined'