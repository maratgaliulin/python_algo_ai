import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from methods.draw_graph import draw_static_graph

PATH_TO_ANALYSIS_DATAFRAMES = "dataframes_for_analysis/"
SAVED_DATAFRAME = 'machine_learning_models/lstm/eurusd/saved_dataframe.csv'
SAVED_PREDICTIONS = 'machine_learning_models/lstm/eurusd/saved_predictions.csv'

eurusd_variables_dict = {
            'SAVED_DATAFRAME': SAVED_DATAFRAME,
            'SAVED_PREDICTIONS': SAVED_PREDICTIONS,
            'AMOUNT_OF_CANDLES': 40
        }


draw_static_graph(
   _t_fr_actual_dir=eurusd_variables_dict['SAVED_DATAFRAME'],
    _t_fr_predicted_dir=eurusd_variables_dict['SAVED_PREDICTIONS'], 
    amount_of_candles=eurusd_variables_dict['AMOUNT_OF_CANDLES']
    )



# print(small_df_raw)

