import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc

def draw_static_graph(
        t_fr_bid:pd.DataFrame,
        t_fr_ask:pd.DataFrame,
        t_fr_joined:pd.DataFrame
        ):
    dot_size = 8

    candlestick_bid = go.Candlestick(
        x=t_fr_bid.index,
        open=t_fr_bid['open'],
        high=t_fr_bid['high'],
        low=t_fr_bid['low'],
        close=t_fr_bid['close'],
        name='Бидовые свечи',
        increasing_line_color='purple',
        decreasing_line_color='darkviolet'
        )
    
    candlestick_ask = go.Candlestick(
        x=t_fr_ask.index,
        open=t_fr_ask['open'],
        high=t_fr_ask['high'],
        low=t_fr_ask['low'],
        close=t_fr_ask['close'],
        name='Асковые свечи',
        increasing_line_color='red',
        decreasing_line_color='pink'
        )
    
    candlestick_joined = go.Candlestick(
        x=t_fr_joined.index,
        open=t_fr_joined['open'],
        high=t_fr_joined['high'],
        low=t_fr_joined['low'],
        close=t_fr_joined['close'],
        name='Объединенные свечи',
        increasing_line_color='green',
        decreasing_line_color='red'
        )
    

    
    fig = make_subplots(rows=1, cols=2, shared_xaxes=False, vertical_spacing=0.02, horizontal_spacing=0.05, row_heights=[1])

    fig.append_trace(candlestick_bid, row=1, col=1)
    fig.append_trace(candlestick_ask, row=1, col=1)
    fig.append_trace(candlestick_joined, row=1, col=2)

    fig.update_xaxes(rangeslider_visible=True)

    app = Dash(__name__)

    app.layout = html.Div( 
        [ 
            html.Div(
            [ 
                dcc.Graph(id='live-update-graph', figure=fig, animate=False, animation_options={'redraw': False, 'duration':50000}, style={"height": "100vh"}), 
                dcc.Interval( id='interval_component', interval=2000), 
            ]), 
        ], 

            style = {"height": "100vh"} ) 
    
    
    app.run(debug=True, use_reloader=False) # Turn off reloader if inside Jupyter

            
            
            
      