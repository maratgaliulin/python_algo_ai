import pandas as pd
import os
import sys

from .functions_offline import get_currency_pairs

project_directory = os.path.abspath(
    os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_directory)

def make_time_series_data(base_directory:str,
                          short_timeframe:str, 
                          long_timeframe:str,
                          time_series:str = '30Min') -> None:
    # Read in the CSV, parse Date and Time into DateTime, then set this as the index of the returned dataframe 
    currency_pairs = get_currency_pairs(base_dir=base_directory)

    # Created a dictionary to tell Pandas how to re-sample, if this isn't in place it will re-sample each column separately
    ohlc_dict = {'Open': 'first', 'High': 'max',
                 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    # Resample mixes the columns so lets re-arrange them
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    for currency_pair in currency_pairs:
        final_dir_bid = base_directory + currency_pair + '/' + short_timeframe + '/Bid/'
        final_dir_ask = base_directory + currency_pair + '/' + short_timeframe + '/Ask/'
        output_dir_bid = base_directory + currency_pair + '/' + long_timeframe + '/Bid/'
        output_dir_ask = base_directory + currency_pair + '/' + long_timeframe + '/Ask/'

        for file in os.listdir(final_dir_bid):
            
            date = os.fsdecode(file).split('_')[5]

            df_bid = pd.read_csv(final_dir_bid + os.fsdecode(file), parse_dates=True, usecols=[
                'Gmt time', 'Open', 'High', 'Low', 'Close', 'Volume'], na_values=['nan'])
            df_bid['Gmt time'] = pd.to_datetime(
                df_bid['Gmt time'], format="%d.%m.%Y %H:%M:%S.%f")
            df_bid.set_index('Gmt time', inplace=True)

            # Resample to amount pointed out in time_series variable
            # (this format is needed) as per ohlc_dict, then remove any line with a NaN

            df_bid = df_bid.resample(time_series, group_keys=True).apply(
                ohlc_dict).dropna(how='any')

            df_bid = df_bid[cols]

            # Write out to CSV
            df_bid.to_csv(output_dir_bid +
                          f"{currency_pair}_{time_series}_Bid_{date}")
            del df_bid

        
        for file in os.listdir(final_dir_ask):
            
            date = os.fsdecode(file).split('_')[5]

            df_ask = pd.read_csv(final_dir_ask + os.fsdecode(file), parse_dates=True, usecols=[
                'Gmt time', 'Open', 'High', 'Low', 'Close', 'Volume'], na_values=['nan'])
            
            df_ask['Gmt time'] = pd.to_datetime(
                df_ask['Gmt time'], format="%d.%m.%Y %H:%M:%S.%f")
            df_ask.set_index('Gmt time', inplace=True)

            # Resample to 15Min (this format is needed) as per ohlc_dict, then remove any line with a NaN
            df_ask = df_ask.resample(time_series, group_keys=True).apply(
                ohlc_dict).dropna(how='any')

            # Write out to CSV
            df_ask.to_csv(output_dir_ask +
                          f"{currency_pair}_{time_series}_Ask_{date}")
            del df_ask

            
            
             
            
            



