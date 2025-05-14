import pandas as pd
import numpy as np

def return_open_price(df:pd.DataFrame, open_bid:str, open_ask:str, close_bid:str):
    if((df[open_ask] != 99999) & (df[open_bid] != 99999)):
        if((df[open_bid] > df[close_bid]) | ((df[open_bid] == df[close_bid]) & (df[open_ask] > df[open_bid]))):        
            return df[open_ask]
        else:
            return df[open_bid]
            
    elif((df[open_bid] != 99999) & (df[open_ask] == 99999)):
        return df[open_bid]
    elif((df[open_bid] == 99999) & (df[open_ask] != 99999)):
        return df[open_ask]
    else:
        return np.nan

def return_close_price(df:pd.DataFrame, close_bid:str, close_ask:str, open_bid:str, open_ask:str):
    if((df[close_bid] != 99999) & (df[close_ask] != 99999)):
        if((df[open_bid] > df[close_bid]) | ((df[open_bid] == df[close_bid]) & (df[open_ask] > df[open_bid]))):        
            return df[close_bid]
        else:
            return df[close_ask]
    
    elif((df[close_ask] != 99999) & (df[close_bid] == 99999)):
        return df[close_ask]
    elif((df[close_ask] == 99999) & (df[close_bid] != 99999)):
        return df[close_bid]
    else:
        return np.nan
    
def return_high_price(df:pd.DataFrame, high_bid:str, high_ask:str):
    if ((df[high_bid] != 99999) & (df[high_ask] != 99999)):
        if((df[high_bid] > df[high_ask])):
            return df[high_bid]
        else:
            return df[high_ask]
    
    elif((df[high_ask] != 99999) & (df[high_bid] == 99999)):
        return df[high_ask]
    elif((df[high_ask] == 99999) & (df[high_bid] != 99999)):
        return df[high_bid]
    else:
        return np.nan
    
def return_low_price(df:pd.DataFrame, low_bid:str, low_ask:str):
    if((df[low_bid] != 99999) & (df[low_ask] != 99999)):
        if((df[low_bid] < df[low_ask])):
            return df[low_bid]
        else:
            return df[low_ask]
    
    elif((df[low_ask] != 99999) & (df[low_bid] == 99999)):
        return df[low_ask]
    elif((df[low_ask] == 99999) & (df[low_bid] != 99999)):
        return df[low_bid]
    else:
        return np.nan