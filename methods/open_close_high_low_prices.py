import pandas as pd

def return_open_price(df:pd.DataFrame):
    if((df['open_ask'] != 999) and (df['open_bid'] != 999)):
        if((df['open_bid'] > df['close_bid']) or ((df['open_bid'] == df['close_bid']) and (df['open_ask'] > df['open_bid']))):        
            return df['open_ask']
        else:
            return df['open_bid']
            
    elif((df['open_bid'] != 999) and (df['open_ask'] == 999)):
        return df['open_bid']
    else:
        return df['open_ask']

def return_close_price(df:pd.DataFrame):
    if((df['close_bid'] != 999) and (df['close_ask'] != 999)):
        if((df['open_bid'] > df['close_bid']) or ((df['open_bid'] == df['close_bid']) and (df['open_ask'] > df['open_bid']))):        
            return df['close_bid']
        else:
            return df['close_ask']
    
    elif((df['close_ask'] != 999) and (df['close_bid'] == 999)):
        return df['close_ask']
    else:
        return df['close_bid']
    
def return_high_price(df:pd.DataFrame):
    if ((df['high_bid'] != 999) and (df['high_ask'] != 999)):
        if((df['high_bid'] > df['high_ask'])):
            return df['high_bid']
        else:
            return df['high_ask']
    
    elif((df['high_ask'] != 999) and (df['high_bid'] == 999)):
        return df['high_ask']
    else:
        return df['high_bid']
    
def return_low_price(df:pd.DataFrame):
    if((df['low_bid'] != 999) and (df['low_ask'] != 999)):
        if((df['low_bid'] < df['low_ask'])):
            return df['low_bid']
        else:
            return df['low_ask']
    
    elif((df['low_ask'] != 999) and (df['low_bid'] == 999)):
        return df['low_ask']
    else:
        return df['low_bid']