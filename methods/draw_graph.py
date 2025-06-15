import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, callback
# import MetaTrader5 as mt
import pytz
from datetime import datetime

def get_first_item_of_list_index(lst:list, li:str):
    new_list = [x for x in lst if x > li]
    if(new_list != []):
        return new_list[0]
    else:
        return lst[-1]

def draw_static_graph(
        _t_fr_actual_dir:str,
        _t_fr_predicted_dir:str, 
        amount_of_candles=10
        ):
    dot_size = 8

    timezone = pytz.timezone('UTC')

    # t_fr_large_test_raw = mt.copy_


# Создание графика объема под свечным графиком
    # fig_vol = go.Bar(
    #     x=_t_fr_small.index, 
    #     y=_t_fr_small['tick_volume'], 
    #     name='График объема',
    #     marker=dict(
    #         color=_t_fr_small['buy_sell_sign'],
    #         colorscale=['red', 'green']
    #     )
    # )

    # Добавление свечей, точек и объема на канвас 

    app = Dash(__name__)

    app.layout = html.Div( 
        [ 
            html.Div(
            [ 
                dcc.Graph(id='live-update-graph', animate=False, animation_options={'redraw': False, 'duration':250000}, style={"height": "100vh"}), 
                dcc.Interval( id='interval_component', interval=250000), 
            ]), 
        ], 

            style = {"height": "100vh"} ) 
    
    @app.callback(
        Output('live-update-graph', 'figure'), 
        [Input('interval_component', 'n_intervals')
    ])

    def update_graph_live(n):

        str_format="%Y-%m-%d %H:%M:%S"

        _t_fr_actual = pd.read_csv(_t_fr_actual_dir, index_col=['time']).sort_index(ascending=True)
        _t_fr_predicted = pd.read_csv(_t_fr_predicted_dir, index_col=['time']).sort_index(ascending=True)
        
        _t_fr_actual = _t_fr_actual.tail(amount_of_candles)
        _t_fr_predicted = _t_fr_predicted.tail(amount_of_candles)
        
        
        
        _t_fr_actual.index = pd.to_datetime(_t_fr_actual.index, format='mixed', errors='raise')
        _t_fr_predicted.index = pd.to_datetime(_t_fr_predicted.index, format='mixed', errors='raise')

        
        candlesticks_actual = go.Candlestick(
            x=_t_fr_actual.index,
            open=_t_fr_actual['open'],
            high=_t_fr_actual['high'],
            low=_t_fr_actual['low'],
            close=_t_fr_actual['close'],
            name='Свечи с сервера',
            increasing_line_color='green',
            decreasing_line_color='red'
            )
        
        candlesticks_predicted = go.Candlestick(
            x=_t_fr_predicted.index,
            open=_t_fr_predicted['open'],
            high=_t_fr_predicted['high'],
            low=_t_fr_predicted['low'],
            close=_t_fr_predicted['close'],
            name='Предсказанные свечи',
            increasing_line_color='orange',
            decreasing_line_color='yellow'
            )
        

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, horizontal_spacing=0.05, column_widths=[1])

        fig.append_trace(candlesticks_actual, row=1, col=1)
        fig.append_trace(candlesticks_predicted, row=1, col=1)
        fig.append_trace(candlesticks_predicted, row=2, col=1)
        
        # fig.add_trace(fig_vol, row=1, col=1)

        # fig.update_layout(template='none')

        fig.update_xaxes(rangeslider_visible=False)
        # fig.update_yaxes(rangemode='tozero')
        fig.update_yaxes()

        return fig
    
    app.run(debug=True, use_reloader=False)

            
            
            
      