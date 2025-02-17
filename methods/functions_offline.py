import os 
import sys 

project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
sys.path.append(project_directory)

from .enums import ENUM_TYPE_OB, ENUM_TYPE_FIND, dataOB
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import enum
import numpy as np
import os
from tqdm import tqdm

# Метод для вычленения необходимого участка малого таймфрейма из всего таймфрейма:

def time_diff_small(df:pd.DataFrame, time_start:str, time_end:str) -> pd.DataFrame:
    ts = pd.to_datetime(time_start)
    te = pd.to_datetime(time_end)
    df_copy = df.loc[ts:te]
    return df_copy

# Метод, позволяющий оценивать впередистоящие свечи для определения начала формирования паттерна для входа в сделку:

def inpFwdStep_calculator(i: int, shft:int, rt:int, dff:int) -> int:
    inpFwdStep = dff if (rt <= shft + i) else i
    return inpFwdStep

# Один из главных методов торговой стратегии. Определяет наличие паттерна для входа в сделку:

def check_pattern(dfr:pd.DataFrame, fwdstep:int, shift:int):  
    '''
    Метод, проверяющий наличие паттерна ордер-блока
    Вводные параметры:
        backstep - число проверяемых свечей, расположенных до текущей свечи (в данном примере - как будет видно ниже при фактической реализации - 0 для первых двух итераций, 2 для последующих)
        fwdstep - число проверяемых свечей, расположенных после текущей свечи (в данном примере - как будет видно ниже при фактической реализации - 0 для последних трех итераций, 3 начиная от первой итерации и далее)
        shift - порядковый номер текущей (исследуемой) итерации таблицы (датафрейма)
    '''    
    i = 1 # инициализация переменной для цикла for
    _val = dfr.iloc[shift]       
    while(i < fwdstep):         # итерация цикла по полному диапазону проверки вокруг исследуемой свечи
        _compare = dfr.iloc[shift + i]
        if(_val['Open'] < _val['Close']):      # условие если текущая свеча восходящая

            '''
            Суть нижеследующей проверки операторов if заключается в ответе на вопрос - имеется ли снятие ликвидности и закрепление телом выше текущей свечи в диапазоне проверки (вокруг свечи) ? 
            При этом согласно стратегии сначала должно произойти снятие ликвидности (СЛ), а затем закрепление (Зк) выше или ниже свечи (если свеча покупная и продажная соответственно)
            Бенчмарк по аномальному объему мы уже проверили выше
            dfr.iloc[shift - i]['low'] - это лой каждой свечи внутри диапазона по итерации в обратном порядке (5я, 4я, 3я итд), _val['low'] - это лой исследуемой свечи (относительно которой проверяются все свечи)
            dfr.iloc[shift - i]['Close'] - цена закрытия каждой свечи внутри диапазона по итерации
            _val['Close'] - цена закрытия исследуемой свечи
            dfr.iloc[shift - i]['Open'] - цена открытия каждой свечи внутри диапазона по итерации
            Снятие ликвидности - это не что иное, как наличие более низкой цены low у какой-л. свечи из диапазона проверки по сравнению с исследуемой свечой (проверяется блоком dfr.iloc[shift - i]['low'] < _val['low'])
            При этом верхняя граница свечи, снимающей ликвидность, не должна быть выше чем верхняя граница исследуемой свечи
            Так как наша свеча покупная, то для того, чтобы было соблюдено условие выше, у свечи из диапазона исследования ни цена открытия, ни цена закрытия не должны
            быть выше цены закрытия нашей текущей свечи - проверяется условиями (dfr.iloc[shift - i]['Close'] < _val['Close']) and (dfr.iloc[shift - i]['Open'] < _val['Close']))
            
            '''

            if (
                ((_compare['Open'] < _compare['Close']) and (_compare['Low'] < _val['Low']) and (_compare['Open'] > _val['Low']) and (_compare['Close'] < _val['High']))
                or
                ((_compare['Open'] > _compare['Close']) and (_compare['Low'] < _val['Low']) and (_compare['Close'] > _val['Low']) and (_compare['Open'] < _val['High']))
                ):
                # Если условия снятия ликвидности соблюдены, то идем дальше. Проверяем закрепление. Оно должно произойти ПОСЛЕ снятия ликвидности, поэтому если СЛ произошло
                # на i-той итерации, нужно ввести новую переменную subtr_val - это остаток от полного диапазона проверки от i-той свечи (на которой сработало условия СЛ) до конца 
                # диапазона. И проверяем остаток диапазона на предмет закрепления. Зк - это когда верхняя граница свечи из диапазона выше верхней границы исследуемой свечи.
                # Т.к. свечи из диапазона могут быть покупными или продажными, то верхней границей их может быть цена открытия или цена закрытия. Проверяем наличие 
                # свеч, у которых цена открытия или цена закрытия выше исследуемой свечи
                j = 0     
                while(i + j < fwdstep):
                    subtr_val = shift + i + j
                    if (
                        (dfr.iloc[subtr_val]['Close'] > _val['High']) 
                        ):                            
                        # если все 3 условия (бенчмарк, СЛ, закрепление) сработали, возвращаем True:
                        total_shift = subtr_val - shift
                        return (True, total_shift)
                    # Если нет, то проверяем дальше до конца диапазона исследования.
                    j += 1                    

        # Относительно красных свеч логика такая же, только в противоположном направлении                        
        elif(_val['Open'] > _val['Close']):  # условие если текущая свеча нисходящая              
            if (
                (_compare['Open'] > _compare['Close']) and (_compare['High'] > _val['High']) and (_compare['Close'] > _val['Low']) and (_compare['Open'] < _val['High'])
                or
                (_compare['Open'] < _compare['Close']) and (_compare['High'] > _val['High']) and (_compare['Close'] < _val['High']) and (_compare['Open'] > _val['Low'])
                ):
                j = 0
                while(i + j < fwdstep):
                    subtr_val = shift + i + j
                    if (
                        (dfr.iloc[subtr_val]['Close'] < _val['Low']) 
                        ):
                        total_shift = subtr_val - shift
                        return (True, total_shift)
                    j += 1
        i+= 1
    
    return False, 1

# Вспомогательный метод. Если при помощи метода check_pattern() был выявлен паттерн для входа в сделку на продажу, осуществляет 
# вход в короткую позицию и постановку стоплосса и тейкпрофита в соответствии с торговой стратегией:

def fibo_sell_order_block(high_price:float, low_price:float, point=0.00001):
    price_diff = high_price - low_price
    entry_point = high_price - 0.5 * price_diff
    stop_loss = low_price + price_diff + point
    take_profit = low_price - 1.2 * price_diff
    
    return entry_point, stop_loss, take_profit

# Вспомогательный метод. Если при помощи метода check_pattern() был выявлен паттерн для входа в сделку на покупку, осуществляет 
# вход в длинную позицию и постановку стоплосса и тейкпрофита в соответствии с торговой стратегией:

def fibo_buy_order_block(low_price:float, high_price:float, point=0.00001):
    price_diff = high_price - low_price
    entry_point = high_price - 0.5 * price_diff
    stop_loss = high_price - price_diff - point
    take_profit = high_price + 1.2 * price_diff
    
    return entry_point, stop_loss, take_profit

# Вспомогательный метод. Выявляет наличие необходимого торгового объема согласно торговой стратегии.  
# Предотвращает вход в позицию, если ТО не соответствует ТС:

def check_benchmark(dfr: pd.DataFrame, coef:float, sft:int, med_vol:float) -> bool:
    benchmark = coef * med_vol
    _val = dfr.iloc[sft]    # просто обозначил текущий таймфрейм чтобы было меньше писанины 
    if(abs(_val['Volume']) > benchmark):    # логика бенчмарка (проверяет если объем торгов в текущей свече больше или равен двукратному медианному объему за исследуемый период). Здесь можно при необходимости вставить более навороченную логику по бенчмарку
        return True
    return False

class EnSearchMode(enum.Enum):   # print(EnSearchMode.Extremum.value)
    Extremum = 0
    Peak = 1
    Bottom = -1

# Вспомогательные методы. Выявляют соответственно высшую и низшую точки цены в рамках указанного временного интервала:

def Lowest(srs: pd.Series, depth: int, start: int):
    if (start < 0):
        return 0
    min_val = srs.iloc[start]
    tstamp = srs.iloc[[start]].index[0]
    for i, val in srs.iloc[start: start + depth].items():
        if (val < min_val):
            min_val = val
            tstamp = i
    return tstamp

def Highest(srs: pd.Series, depth: int, start: int):
    if (start < 0):
        return 0
    max_val = srs.iloc[start]
    tstamp = srs.iloc[[start]].index[0]
    for i, val in srs.iloc[start: start + depth].items():
        if (val > max_val):
            max_val = val
            tstamp = i
    return tstamp

# Вспомогательный метод.

def zz_calc_wo_graph(df:pd.DataFrame, InpDepth:int = 5, BackStep:int = 3):
    rates_total = len(df)
    start = 0
    shift = 0
    last_high_pos = 0.0
    last_low_pos = 0.0
    val = 0.0
    last_high = 0.0
    last_low = 0.0
    high = df['High']
    low = df['Low']
    HighMapBuffer = np.zeros_like(high)
    LowMapBuffer = np.zeros_like(low)
    ZigZagBuffer = np.zeros_like(low)
    extreme_search = np.zeros_like(low)
    start = 0

    shift = start

    while (shift < rates_total):
        # search for low values:
        val = low.loc[Lowest(low, depth=InpDepth, start=shift)]
        InpBackstep = 0 if shift <= BackStep else shift - BackStep
        last_low = low.loc[Lowest(low, depth=InpDepth, start=InpBackstep)]
        if (val >= last_low):
            val = 0.0
        else:
            last_low = val

        if (low.iloc[shift] <= last_low):
            LowMapBuffer[shift] = low.iloc[shift]
            extreme_search[shift] = EnSearchMode.Bottom.value
        else:
            LowMapBuffer[shift] = 0.0

        # search for high values:
        val = high[Highest(high, InpDepth, shift)]
        last_high = high[Highest(high, InpDepth, start=InpBackstep)]
        if (val <= last_high):
            val = 0.0
        else:
            last_high = val

        if (high.iloc[shift] >= last_high):
            HighMapBuffer[shift] = high.iloc[shift]
            extreme_search[shift] = EnSearchMode.Peak.value
        else:
            HighMapBuffer[shift] = 0.0

        shift += 1
        # print(last_low, last_high)

    extreme_search = np.int16(extreme_search)

    shift = start


    while (shift < rates_total):
        res = 0.0
        match(extreme_search[shift]):
            case EnSearchMode.Extremum.value:
                if (last_low == 0.0 and last_high == 0.0):
                    if (HighMapBuffer[shift] != 0):
                        last_high = high.iloc[shift]
                        last_high_pos = shift
                        extreme_search[shift] = EnSearchMode.Bottom.value
                        ZigZagBuffer[shift] = last_high
                        res = 1
                    if (LowMapBuffer[shift] != 0):
                        last_low = low.iloc[shift]
                        last_low_pos = shift
                        extreme_search[shift] = EnSearchMode.Peak.value
                        ZigZagBuffer[shift] = last_low
                        res = 1
                # break
            case EnSearchMode.Peak.value:
                if (LowMapBuffer[shift] != 0.0 and HighMapBuffer[shift] == 0.0):
                    ZigZagBuffer[int(last_low_pos)] = 0.0
                    last_low_pos = shift
                    last_low = LowMapBuffer[shift]
                    ZigZagBuffer[shift] = last_low
                    res = 1
                if (HighMapBuffer[shift] != 0.0 and LowMapBuffer[shift] == 0.0):
                    last_high = HighMapBuffer[shift]
                    last_high_pos = shift
                    ZigZagBuffer[shift] = last_high
                    extreme_search[shift] = EnSearchMode.Bottom.value
                    res = 1
                # break
            case EnSearchMode.Bottom.value:
                if (HighMapBuffer[shift] != 0.0 and LowMapBuffer[shift] == 0.0):
                    ZigZagBuffer[int(last_high_pos)] = 0.0
                    last_high_pos = shift
                    last_high = HighMapBuffer[shift]
                    ZigZagBuffer[shift] = last_high
                if (LowMapBuffer[shift] != 0.0 and HighMapBuffer[shift] == 0.0):
                    last_low = LowMapBuffer[shift]
                    last_low_pos = shift
                    ZigZagBuffer[shift] = last_low
                    extreme_search[shift] = EnSearchMode.Peak.value
                # break
            case _:
                print(rates_total)
        shift += 1


    zzbuff_series = pd.Series(data=ZigZagBuffer, index=df.index)
    highmapbuff_series = pd.Series(data=HighMapBuffer, index=df.index)
    lowmapbuff_series = pd.Series(data=LowMapBuffer, index=df.index)
    
    return zzbuff_series[zzbuff_series != 0.0], highmapbuff_series[highmapbuff_series != 0.0], lowmapbuff_series[lowmapbuff_series != 0.0]

# Один из главных методов. Действует вместе с методом fibo_sell_order_block(), в рамках той же цели:

def sell_order_block(dfr:pd.DataFrame, time_start_narrow:str, t_fr_small=1, input_depth=5, back_step=4):
    time_end_narrow = pd.to_datetime(time_start_narrow) + pd.Timedelta(hours=6)
    d_fr_narrow = time_diff_small(dfr, time_start_narrow, time_end_narrow)
    zz_buff, highmapbuff, lowmapbuff = zz_calc_wo_graph(df=d_fr_narrow, InpDepth=input_depth, BackStep=back_step)
    
    i = 0
    j = 0

    try:
        while(
            (lowmapbuff.iloc[i] < lowmapbuff.iloc[i-1])
            or
            (lowmapbuff.iloc[i+1] < lowmapbuff.iloc[i])
            or
            (lowmapbuff.iloc[i+2] < lowmapbuff.iloc[i])
        ):
            i+=1

        while (
            (highmapbuff.iloc[j] < highmapbuff.iloc[j-1])
            or
            (highmapbuff.iloc[j+1] < highmapbuff.iloc[j])
            or
            (highmapbuff.iloc[j+2] < highmapbuff.iloc[j])
        ):
            j += 1


        if(i>=j):
            first_high, first_high_index, first_low, first_low_index = highmapbuff.iloc[
                i], highmapbuff.index[i], lowmapbuff.iloc[i], lowmapbuff.index[i]
            hmb = highmapbuff.iloc[i:]
            lmb = lowmapbuff.iloc[i:]
        else:
            first_high, first_high_index, first_low, first_low_index = highmapbuff.iloc[
                j], highmapbuff.index[j], lowmapbuff.iloc[j], lowmapbuff.index[j]
            hmb = highmapbuff.iloc[j:]
            lmb = lowmapbuff.iloc[j:]


        k = 0
        m = 0

        while (
            (lmb.iloc[k] > lmb.iloc[k-1])
            or
            (lmb.iloc[k+1] > lmb.iloc[k])
            or
            (lmb.iloc[k+2] > lmb.iloc[k])
        ):
            k+=1


        while (
            (hmb.iloc[m] > hmb.iloc[m-1])
            or
            (hmb.iloc[m+1] > hmb.iloc[m])
            or
            (hmb.iloc[m+2] > hmb.iloc[m])
        ):
            m += 1


        if (k >= m):
            first_high, first_high_index, first_low, first_low_index = hmb.iloc[
                k], hmb.index[k], lmb.iloc[k], lmb.index[k]
            hmb = hmb.iloc[k:]
            lmb = lmb.iloc[k:]
        else:
            first_high, first_high_index, first_low, first_low_index = hmb.iloc[
                m], hmb.index[m], lmb.iloc[m], lmb.index[m]
            hmb = hmb.iloc[m:]
            lmb = lmb.iloc[m:]

        dfr_intermed = d_fr_narrow.loc[first_high_index:first_low_index]

        if not dfr_intermed.empty:

            final_high, final_high_index, bos_point, bos_point_index = dfr_intermed['High'].max(), dfr_intermed['High'].idxmax(), dfr_intermed['Low'].min(), dfr_intermed['Low'].idxmin()

            dfr_final = time_diff_small(d_fr_narrow, final_high_index, time_end_narrow)

            zz_buff, highmapbuff, lowmapbuff = zz_calc_wo_graph(df=dfr_final, BackStep=4)

            cur_low, cur_low_idx = lowmapbuff.iloc[0], lowmapbuff.index[0]
            init_diff = final_high - lowmapbuff.iloc[0]
            entry_point_initiated = False
            while(
                i < len(dfr_final)
            ):
                if(
                    dfr_final.iloc[i]['Low'] <= cur_low
                ):
                    cur_low = dfr_final.iloc[i]['Low']
                
                entry, sl, tp = fibo_sell_order_block(final_high, cur_low)

                if(
                    dfr_final.iloc[i]['High'] >= entry
                    or
                    dfr_final.iloc[i]['Open'] >= entry
                    or
                    dfr_final.iloc[i]['Close'] >= entry
                ):
                    # print(f"sell entry point initiated: {entry}, stoploss: {sl}, takeprofit: {tp}, entry point index: {dfr_final.index[i]}")

                    analysis_df_row = {
                        'time': dfr_final.index[i], 
                        'type': 'entry',
                        'direction': 'sell',
                        'entry': entry,
                        'take_profit': tp, 
                        'stop_loss': sl,
                        'profit_amount': entry - tp, 
                        'profit_percentage': (entry - tp) / entry * 100,
                        'loss_amount': sl - entry,
                        'loss_percentage': (sl - entry) / entry * 100
                    }

                    dfr_final = dfr_final.loc[dfr_final.index[i]:]
                    entry_point_initiated = True
                    break

                if(
                    final_high - cur_low >= init_diff * 4
                ):
                    dfr_final = dfr_final.loc[dfr_final.index[i]:]
                    t_delta = int(pd.Timedelta(dfr_final.index[i] - pd.to_datetime(time_start_narrow)).total_seconds() / 60 / 30)
                    return t_delta, analysis_df_row
                i+=1

            j = 0
            if(
                entry_point_initiated == True
            ):
                while (j < len(dfr_final)-1):
                    if(
                        dfr_final.iloc[j]['High'] >= sl
                        or
                        dfr_final.iloc[j]['Open'] >= sl
                        or
                        dfr_final.iloc[j]['Close'] >= sl
                    ):
                        # print(f"sell stoploss activated: {sl}, stoploss_index: {dfr_final.index[j]}")

                        analysis_df_row = {
                            'time': dfr_final.index[j], 
                            'type': 'stop_loss',
                            'direction': 'sell',
                            'entry': entry,
                            'take_profit': 0.0, 
                            'stop_loss': sl,
                            'profit_amount': 0.0, 
                            'profit_percentage': 0.0,
                            'loss_amount': sl - entry,
                            'loss_percentage': (sl - entry) / entry * 100
                            }

                        break
                        
                    elif(
                        dfr_final.iloc[j]['Low'] <= tp
                        or
                        dfr_final.iloc[j]['Open'] <= tp
                        or
                        dfr_final.iloc[j]['Close'] <= tp
                    ):
                        # print(f"sell takeprofit activated: {tp}, takeprofit_index: {dfr_final.index[j]}")

                        analysis_df_row = {
                            'time': dfr_final.index[j], 
                            'type': 'take_profit',
                            'direction': 'sell',
                            'entry': entry,
                            'take_profit': tp, 
                            'stop_loss': 0.0,
                            'profit_amount': entry - tp, 
                            'profit_percentage': (entry - tp) / entry * 100,
                            'loss_amount': 0.0,
                            'loss_percentage': 0.0
                            }
                        

                        break
                    j+=1

            t_delta = int(pd.Timedelta(dfr_final.index[j] - pd.to_datetime(time_start_narrow)).total_seconds() / 60 / 30)

            return t_delta, analysis_df_row
        else:
            return 1, {}
    except:
        return 1, {}

# Один из главных методов. Действует вместе с методом fibo_buy_order_block(), в рамках той же цели:

def buy_order_block(dfr:pd.DataFrame, time_start_narrow:str, t_fr_small=1, input_depth=5, back_step=4):

    time_end_narrow = pd.to_datetime(time_start_narrow) + pd.Timedelta(hours=6)
    d_fr_narrow = time_diff_small(dfr, time_start_narrow, time_end_narrow)
    zz_buff, highmapbuff, lowmapbuff = zz_calc_wo_graph(df=d_fr_narrow, InpDepth=input_depth, BackStep=back_step)
    
    i = 0
    j = 0

    try:
        while(
            (lowmapbuff.iloc[i] > lowmapbuff.iloc[i-1])
            or
            (lowmapbuff.iloc[i+1] > lowmapbuff.iloc[i])
            or
            (lowmapbuff.iloc[i+2] > lowmapbuff.iloc[i])):
            i+=1

        while ((highmapbuff.iloc[j] > highmapbuff.iloc[j-1])
            or
            (highmapbuff.iloc[j+1] > highmapbuff.iloc[j])
            or
            (highmapbuff.iloc[j+2] > highmapbuff.iloc[j])
            ):
            j += 1



        if(i>=j):
            first_high, first_high_index, first_low, first_low_index = highmapbuff.iloc[i], highmapbuff.index[i], lowmapbuff.iloc[i], lowmapbuff.index[i]
            hmb = highmapbuff.iloc[i:]
            lmb = lowmapbuff.iloc[i:]
        else:
            first_high, first_high_index, first_low, first_low_index = highmapbuff.iloc[j], highmapbuff.index[j], lowmapbuff.iloc[j], lowmapbuff.index[j]
            hmb = highmapbuff.iloc[j:]
            lmb = lowmapbuff.iloc[j:]


        k = 0
        m = 0

        while (
            (lmb.iloc[k] < lmb.iloc[k-1])
            or
            (lmb.iloc[k+1] < lmb.iloc[k])
            or
            (lmb.iloc[k+2] < lmb.iloc[k])
        ):
            k+=1


        while (
            (hmb.iloc[m] < hmb.iloc[m-1])
            or
            (hmb.iloc[m+1] < hmb.iloc[m])
            or
            (hmb.iloc[m+2] < hmb.iloc[m])
        ):
            m += 1


        if (k >= m):
            first_high, first_high_index, first_low, first_low_index = hmb.iloc[k], hmb.index[k], lmb.iloc[k], lmb.index[k]
            hmb = hmb.iloc[k:]
            lmb = lmb.iloc[k:]
        else:
            first_high, first_high_index, first_low, first_low_index = hmb.iloc[m], hmb.index[m], lmb.iloc[m], lmb.index[m]
            hmb = hmb.iloc[m:]
            lmb = lmb.iloc[m:]

        dfr_intermed = d_fr_narrow.loc[first_high_index:first_low_index]

        if not dfr_intermed.empty:

            final_low, final_low_index, bos_point, bos_point_index = dfr_intermed['Low'].min(), dfr_intermed['Low'].idxmin(), dfr_intermed['Low'].max(), dfr_intermed['Low'].idxmax()
            dfr_final = time_diff_small(d_fr_narrow, final_low_index, time_end_narrow)

            zz_buff, highmapbuff, lowmapbuff = zz_calc_wo_graph(df=dfr_final, BackStep=4)


            cur_high, cur_high_idx = highmapbuff.iloc[0], highmapbuff.index[0]
            init_diff = highmapbuff.iloc[0] - final_low
            entry_point_initiated = False
            while(
                i < len(dfr_final)
            ):
                if(
                    dfr_final.iloc[i]['High'] >= cur_high
                ):
                    cur_high = dfr_final.iloc[i]['High']
                
                entry, sl, tp = fibo_buy_order_block(final_low, cur_high)

                # print(entry, sl, tp)

                if(
                    dfr_final.iloc[i]['Low'] <= entry
                    or
                    dfr_final.iloc[i]['Open'] <= entry
                    or
                    dfr_final.iloc[i]['Close'] <= entry
                ):
                    # print(f"buy entry point initiated: {entry}, stoploss: {sl}, takeprofit: {tp}, entry point index: {dfr_final.index[i]}")

                    analysis_df_row = {
                        'time': dfr_final.index[i], 
                        'type': 'entry',
                        'direction': 'buy',
                        'entry': entry,
                        'take_profit': tp, 
                        'stop_loss': sl,
                        'profit_amount': tp - entry, 
                        'profit_percentage': (tp - entry) / entry * 100,
                        'loss_amount': entry - sl,
                        'loss_percentage': (entry - sl) / entry * 100
                    }


                    dfr_final = dfr_final.loc[dfr_final.index[i]:]
                    entry_point_initiated = True
                    break

                if(
                    cur_high - final_low >= init_diff * 4
                ):
                    dfr_final = dfr_final.loc[dfr_final.index[i]:]
                    t_delta = int(pd.Timedelta(dfr_final.index[i] - pd.to_datetime(time_start_narrow)).total_seconds() / 60 / 30)
                    return t_delta, analysis_df_row
                i+=1

            j = 0
            if(
                entry_point_initiated == True
            ):
                while (j < len(dfr_final)-1):
                    if(
                        dfr_final.iloc[j]['Low'] <= sl
                        or
                        dfr_final.iloc[j]['Open'] <= sl
                        or
                        dfr_final.iloc[j]['Close'] <= sl
                    ):
                        # print(f"buy stoploss activated: {sl}, stoploss_index: {dfr_final.index[j]}")

                        analysis_df_row = {
                            'time': dfr_final.index[j], 
                            'type': 'stop_loss',
                            'direction': 'buy',
                            'entry': entry,
                            'take_profit': 0.0, 
                            'stop_loss': sl,
                            'profit_amount': 0.0, 
                            'profit_percentage': 0.0,
                            'loss_amount': entry - sl,
                            'loss_percentage': (entry - sl) / entry * 100
                        }
                        
                        break
                        
                        
                    elif(
                        dfr_final.iloc[j]['High'] >= tp
                        or
                        dfr_final.iloc[j]['Open'] >= tp
                        or
                        dfr_final.iloc[j]['Close'] >= tp
                    ):
                        # print(f"buy takeprofit activated: {tp}, takeprofit_index: {dfr_final.index[j]}")

                        analysis_df_row = {
                            'time': dfr_final.index[j], 
                            'type': 'take_profit',
                            'direction': 'buy',
                            'entry': entry,
                            'take_profit': tp, 
                            'stop_loss': 0.0,
                            'profit_amount': tp - entry, 
                            'profit_percentage': (tp - entry) / entry * 100,
                            'loss_amount': 0.0,
                            'loss_percentage': 0.0
                        }
                        

                        break
                    j+=1

            t_delta = int(pd.Timedelta(dfr_final.index[j] - pd.to_datetime(time_start_narrow)).total_seconds() / 60 / 30)

            return t_delta, analysis_df_row
        else:
            return 1, {}
    except:
        return 1, {}

# Метод, позволяющий получить валютные пары из директории:

def get_currency_pairs(base_dir = '../hist_data/') -> list:
    currency_pairs = []
    for file in os.listdir(base_dir):
        currency_pairs.append(os.fsdecode(file))
    return currency_pairs

# Вспомогательный метод. Объединяет исторические данные по конкретному инструменту, которые распололжены в разных файлах, в
# один общий датафрейм, который впоследствии возвращает.

def get_hist_data(currency_pair:str, timeframe:str, bid_or_ask:str) -> pd.DataFrame:
    
    directory = '../hist_data/' + currency_pair + '/' + timeframe + '/' + bid_or_ask
    print('Parsing from directory:', directory)
    final_dir = os.fsencode(directory)
    
    file_list = []
    df_test_total = pd.DataFrame()
    
    for file in os.listdir(final_dir):
        file_list.append(os.fsdecode(final_dir) + '/' + os.fsdecode(file))

    for file in file_list:
        try:
            df_test_total = pd.concat([df_test_total, pd.read_csv(file, index_col='Gmt time')])
        except:
            pass
        
    try:
        df_test_total.index = pd.to_datetime(
            df_test_total.index, format="%d.%m.%Y %H:%M:%S.%f")
    except:
        df_test_total.index = pd.to_datetime(
            df_test_total.index, format="%Y-%m-%d %H:%M:%S")

    df_test_total.sort_index(ascending=True, inplace=True)
        
    return df_test_total

# Самый главный метод, который объединяет все вышеперечисленные методы и непосредственно "решает", когда необходимо входить в позицию,
# на какие цены ставить стоплосс и тейкпрофит итд. В результате своей работы возвращает датафрейм, в котором записаны такие параметры 
# как: дата/время, цена входа в позицию, цена стоплосса, цена тейкпрофита и др.:

def buy_or_sell(large_bid:pd.DataFrame, large_ask:pd.DataFrame, small_bid:pd.DataFrame, small_ask:pd.DataFrame, input_depth=5, back_step=4) -> pd.DataFrame:
    rates_total = len(large_bid)
    
    low = large_bid['Low']
    # volume = large_bid['Volume']
    OBBuffer = np.zeros_like(low)
    zakrepBuffer = np.zeros_like(low)
    shift = 0
    analysis_df = pd.DataFrame(columns=['time', 'type', 'direction', 'entry', 'take_profit', 'stop_loss', 'profit_amount', 'profit_percentage', 'loss_amount', 'loss_percentage'])
    # median_vol = volume.median() 
    ob_calculation_candles = 1
    
    pbar = tqdm(total=rates_total+1, iterable=shift)
    while (shift < rates_total):
        diff = rates_total - shift
        InpFwdstep = inpFwdStep_calculator(input_depth, shift, rates_total, diff)
        # if(check_benchmark(df_30_min, 0, shift, median_vol) is True): 
        ifTrue, sval = check_pattern(large_bid, InpFwdstep, shift)     
        try:
            if((ifTrue is True) and (large_bid.iloc[shift]['Open'] < large_bid.iloc[shift]['Close'])):
                # OBBuffer - numpy-массив, равный длине нашего датафрейма, исходно заполненный нулями (см.выше в списке всех переменных), который при срабатывании условий заполняется средним значением цены в данной свече
                # Я выбрал среднее значение цены, чтобы красные точки, обозначающие свечи в которых сработали условия ордер-блока, располагались посередине данных свеч (для наглядности)
                OBBuffer[shift] = large_bid.iloc[shift]['Low'] # здесь будет логика перехода на 1-минутный таймфрейм 
                

                ob_calculation_candles, analysis_df_row = buy_order_block(small_bid, large_bid.index[shift + sval], input_depth=input_depth, back_step=back_step)
                
                if(analysis_df_row != {}):
                    df_dictionary = pd.DataFrame([analysis_df_row])
                    analysis_df = pd.concat([analysis_df, df_dictionary], ignore_index=True)
                        
                    # if(large_bid.iloc[shift + sval]['Close'] > large_bid.iloc[shift + sval]['Open']):
                    #     zakrepBuffer[shift + sval] = large_bid.iloc[shift + sval]['High']
                    # else:
                    #     zakrepBuffer[shift + sval] = large_bid.iloc[shift + sval]['Low']
                    
                    
                    
            elif((ifTrue is True) and (large_ask.iloc[shift]['Open'] > large_ask.iloc[shift]['Close'])):  
                OBBuffer[shift] = large_ask.iloc[shift]['High']
                ob_calculation_candles, analysis_df_row = sell_order_block(small_ask, large_ask.index[shift + sval], input_depth=input_depth, back_step=back_step)
                if(analysis_df_row != {}):
                    df_dictionary = pd.DataFrame([analysis_df_row])
                    analysis_df = pd.concat([analysis_df, df_dictionary], ignore_index=True)
                    
                    # (analysis_df.copy() if df_dictionary.empty 
                    #                else df_dictionary.copy() if analysis_df.empty 
                                
                    #                else pd.concat([analysis_df, df_dictionary]) )
                    
                    # 
                    
                    # if(large_ask.iloc[shift + sval]['Close'] > large_ask.iloc[shift + sval]['Open']):
                    #     zakrepBuffer[shift + sval] = large_ask.iloc[shift + sval]['High']
                    # else:
                    #     zakrepBuffer[shift + sval] = large_ask.iloc[shift + sval]['Low']

            if(ob_calculation_candles >= sval):
                shift += ob_calculation_candles
                pbar.update(ob_calculation_candles)
            else:
                shift += 1
                pbar.update(1)
        except:
            shift +=1
            pbar.update(1)
    pbar.close()

    # large_bid['ob_price'] = OBBuffer  # создание в датафрейме колонки ob_price из массива OBBuffer (нужно для того, чтобы каждый элемент данного массива получил индекс в виде даты-времени из датафрейма)
    # large_bid['zakrep_price'] = zakrepBuffer    
    analysis_df.set_index('time', inplace=True)
    analysis_df.sort_index(ascending=True, inplace=True)

    return analysis_df

# Вспомогательные методы, которые обеспечивают автоматизацию сбора данных по торговым инструментам в единый датафрейм для последующего 
# анализа и графического представления:

def analysis_df_to_csv(input_backsteps:list, input_depths:list) -> None:
    currency_pairs = get_currency_pairs()
    tf_1_min = '1_min'
    tf_30_min = '30_min'
    ask = 'Ask'
    bid = 'Bid'
    bt_results = os.listdir('backtest_results')
    
    for currency_pair in currency_pairs:
        df_eurusd_large_bid = get_hist_data(currency_pair, tf_30_min, bid)
        df_eurusd_large_ask = get_hist_data(currency_pair, tf_30_min, ask)
        df_eurusd_small_bid = get_hist_data(currency_pair, tf_1_min, bid)
        df_eurusd_small_ask = get_hist_data(currency_pair, tf_1_min, ask)
        
        for inputDepth in input_depths:
            for inputBackStep in input_backsteps:
                backtest_file_name = f'{currency_pair}_inp_depths_{inputDepth}_backstep_{inputBackStep}_backtest.csv'
                if(backtest_file_name not in bt_results):
                    buy_or_sell_df = buy_or_sell(df_eurusd_large_bid, df_eurusd_large_ask, df_eurusd_small_bid, df_eurusd_small_ask, input_depth=inputDepth, back_step=inputBackStep)
                    # print(buy_or_sell_df.head())
                    buy_or_sell_df.to_csv(f'backtest_results/{currency_pair}_inp_depths_{inputDepth}_backstep_{inputBackStep}_backtest.csv', index='time')
                    del buy_or_sell_df

def single_backtest_result_analysis(file_name:str) -> dict:
    base_dir = 'backtest_results/'
    
    df = pd.read_csv(base_dir + file_name)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    
    file_name_list = file_name.strip('.csv').split('_')
    currency_pair = file_name_list[0]
    inpDepth = file_name_list[3]
    inpBackstep = file_name_list[5]
    row_name = currency_pair + '_depth_' + inpDepth + '_bstp_' + inpBackstep
    shoulder_50 = 50
    shoulder_100 = 100
    deposit = 1000
    comission = 0.0003
    our_percentage = 0.3
    
    df['TP_shoulder_50'] = df['profit_amount'].apply(lambda x: round(x * shoulder_50 * deposit - deposit * shoulder_50 * comission, 3) if x > 0 else 0.0)
    df['SL_shoulder_50'] = df['loss_amount'].apply(lambda x: round(x * shoulder_50 * deposit + deposit * shoulder_50 * comission, 3) if x > 0 else 0.0)
    
    df['TP_shoulder_100'] = df['profit_amount'].apply(lambda x: round(x * shoulder_100 * deposit - deposit * shoulder_100 * comission, 3) if x > 0 else 0.0)
    df['SL_shoulder_100'] = df['loss_amount'].apply(lambda x: round(x * shoulder_100 * deposit + deposit * shoulder_100 * comission, 3) if x > 0 else 0.0)
    
    total_profit_50_shoulder = round(df['TP_shoulder_50'].sum(), 3)
    total_loss_50_shoulder = round(df['SL_shoulder_50'].sum(), 3)
    
    total_profit_50_shoulder_per_year = round(total_profit_50_shoulder / 20,3)
    total_loss_50_shoulder_per_year = round(total_loss_50_shoulder / 20,3)
    
    median_profit_50_shoulder = round(df['TP_shoulder_50'].mean(),3)
    median_loss_50_shoulder = round(df['SL_shoulder_50'].mean(),3)
    
    profit_to_loss_50_ratio = round(total_profit_50_shoulder / total_loss_50_shoulder,3)
    median_annual_margin_50 = round(total_profit_50_shoulder_per_year - total_loss_50_shoulder_per_year,3)
    our_margin_50_per_year = round(median_annual_margin_50 * our_percentage,3)
    
    
    try:
        if median_profit_50_shoulder != 0.0:            
            margin_alicount_50 = round((total_profit_50_shoulder - total_loss_50_shoulder),3)
        else:
            margin_alicount_50 = 0.0
    except:
        margin_alicount_50 = 0.0
        
    try:
        if median_profit_50_shoulder != 0.0: 
            total_profit_to_median_profit_ratio_50 = round(total_profit_50_shoulder / median_profit_50_shoulder,3)
        else:
            total_profit_to_median_profit_ratio_50 = 0.0
    except:
        total_profit_to_median_profit_ratio_50 = 0.0
        
        
    try:
        if median_profit_50_shoulder != 0.0: 
            median_profit_to_median_loss_ratio_50 = round(median_profit_50_shoulder / median_loss_50_shoulder,3)
        else:
            median_profit_to_median_loss_ratio_50 = 0.0
        
    except:        
        median_profit_to_median_loss_ratio_50 = 0.0
        
    margin_alicount_50_per_year = round(margin_alicount_50 / 20,3)
    
    
    total_profit_100_shoulder = round(df['TP_shoulder_100'].sum(),3)
    total_loss_100_shoulder = round(df['SL_shoulder_100'].sum(),3)
    
    total_profit_100_shoulder_per_year = round(total_profit_100_shoulder / 20,3)
    total_loss_100_shoulder_per_year = round(total_loss_100_shoulder / 20,3)
    
    median_profit_100_shoulder = round(df['TP_shoulder_100'].mean(),3)
    median_loss_100_shoulder = round(df['SL_shoulder_100'].mean(),3)
    
    profit_to_loss_100_ratio = round(total_profit_100_shoulder / total_loss_100_shoulder,3)
    median_annual_margin_100 = round(total_profit_100_shoulder_per_year - total_loss_100_shoulder_per_year,3)
    our_margin_100_per_year = round(median_annual_margin_100 * our_percentage,3)
    
    try:
        if median_profit_100_shoulder != 0.0:
            margin_alicount_100 = round((total_profit_100_shoulder - total_loss_100_shoulder),3)
        else:
            margin_alicount_100 = 0.0
    except:
        margin_alicount_100 = 0.0
        
    try:
        if median_profit_100_shoulder != 0.0:
            total_profit_to_median_profit_ratio_100 = round(total_profit_100_shoulder / median_profit_100_shoulder,3)
        else:
            total_profit_to_median_profit_ratio_100 = 0.0
    except:
        total_profit_to_median_profit_ratio_100 = 0.0
        
    try:
        if median_profit_100_shoulder != 0.0:
            median_profit_to_median_loss_ratio_100 = round(median_profit_100_shoulder / median_loss_100_shoulder,3)
        else:
            median_profit_to_median_loss_ratio_100 = 0.0
    except:
        median_profit_to_median_loss_ratio_100 = 0.0
        
    margin_alicount_100_per_year = round(margin_alicount_100 / 20,3)
    
    amount_of_deals_per_year = len(df)
    
    output_dict_1_50 = {
        'Вал.пара_глубина_шаг': row_name,
        'Кол-во сделок в год':amount_of_deals_per_year,
        'Общая выручка':total_profit_50_shoulder,
        'Общий убыток':total_loss_50_shoulder,
        'Общ.выручка/год': total_profit_50_shoulder_per_year,
        'Общ.убыт./год': total_loss_50_shoulder_per_year,
        'Средняя выручка': median_profit_50_shoulder,
        'Средний убыток':median_loss_50_shoulder,
        'Отношение приб.к уб.':profit_to_loss_50_ratio,
        'Сред.ежегод.чист.приб.':median_annual_margin_50,
        'Наша ежегод.выручка':our_margin_50_per_year,
        'Отн.общей прибыли к средней прибыли':total_profit_to_median_profit_ratio_50,
        'Отн.сред.выручки к сред.убытку':median_profit_to_median_loss_ratio_50,
        'Отн.чистой прибыли к ср.выручке':margin_alicount_50,
        'Ср.чист.приб.клиента в год с $1,000':margin_alicount_50_per_year,
    }
        
    output_dict_1_100 = {
        'Вал.пара_глубина_шаг': row_name,
        'Кол-во сделок в год':amount_of_deals_per_year,
        'Общая выручка':total_profit_100_shoulder,
        'Общий убыток':total_loss_100_shoulder,
        'Общ.выручка/год': total_profit_100_shoulder_per_year,
        'Общ.убыт./год': total_loss_100_shoulder_per_year,
        'Средняя выручка': median_profit_100_shoulder,
        'Средний убыток':median_loss_100_shoulder,
        'Отношение приб.к уб.':profit_to_loss_100_ratio,
        'Сред.ежегод.чист.приб.':median_annual_margin_100,
        'Наша ежегод.выручка':our_margin_100_per_year,
        'Отн.общей прибыли к средней прибыли':total_profit_to_median_profit_ratio_100,
        'Отн.сред.выручки к сред.убытку':median_profit_to_median_loss_ratio_100,
        'Отн.чистой прибыли к ср.выручке':margin_alicount_100,
        'Ср.чист.приб.клиента в год с $1,000':margin_alicount_100_per_year,
    }
    
    return output_dict_1_50, output_dict_1_100

def multiple_backtests_results_analysis() -> None:
    
    
    currency_pairs = get_currency_pairs()
    base_dir = 'backtest_results/'
    output_dir = 'backtest_analytics/'
    
    output_dir_50 = output_dir + 'shoulder_50/'
    output_dir_100 = output_dir + 'shoulder_100/'
    
    raw_data = os.listdir(base_dir)
    
    cols = ['Вал.пара_глубина_шаг','Кол-во сделок в год','Общая выручка','Общий убыток','Общ.выручка/год','Общ.убыт./год','Средняя выручка','Средний убыток','Отношение приб.к уб.','Сред.ежегод.чист.приб.','Наша ежегод.выручка','Отн.общей прибыли к средней прибыли','Отн.сред.выручки к сред.убытку','Отн.чистой прибыли к ср.выручке','Ср.чист.приб.клиента в год с $1,000']
    
    for currency_pair in currency_pairs:
        df_50 = pd.DataFrame(columns=cols)
        df_100 = pd.DataFrame(columns=cols)
        # print(df_50, df_100)
        
        for file_name in raw_data:
            file_name_to_list = file_name.strip('.csv').split('_')
            
            if(currency_pair == file_name_to_list[0]):
                row_df_50, row_df_100 = single_backtest_result_analysis(file_name=file_name)
                single_row_50 = pd.DataFrame([row_df_50])
                single_row_100 = pd.DataFrame([row_df_100])
                # print(single_row_50)
                # print(single_row_100)
                df_50 = (df_50.copy() if single_row_50.empty else single_row_50.copy() if df_50.empty
                        else pd.concat([df_50, single_row_50], ignore_index=True) # if both DataFrames non empty
                        )
                
                df_100 = (df_100.copy() if single_row_100.empty else single_row_100.copy() if df_100.empty
                        else pd.concat([df_100, single_row_100], ignore_index=True) # if both DataFrames non empty
                        )
                
                df_50.to_csv(f'{output_dir_50 + file_name_to_list[0]}_1_50.csv')
                df_100.to_csv(f'{output_dir_100 + file_name_to_list[0]}_1_100.csv')
        # del df_50, df_100

def draw_single_currency_pair_backtest_analytics(currency_pair:str, save_html=False, save_img=False)->None:
    base_dir = 'backtest_analytics/'
    shoulder_50 = 'shoulder_50/'
    shoulder_100 = 'shoulder_100/'
    file_name_shoulder_50 = currency_pair + '_1_50.csv'
    file_name_shoulder_100 = currency_pair + '_1_100.csv'
    
    backtest_html_dir = 'backtest_result_graphs_html/'
    backtest_img_dir = 'backtest_result_graphs_img/'

    file_dir_shoulder_50 = base_dir + shoulder_50 + file_name_shoulder_50
    file_dir_shoulder_100 = base_dir + shoulder_100 + file_name_shoulder_100

    
    cols = ['Вал.пара_глубина_шаг', 
    "Кол-во сделок в год", 
    "Общая выручка", 
    "Общий убыток", 
    "Общ.выручка/год", 
    "Общ.убыт./год", 
    "Средняя выручка", 
    "Средний убыток", 
    "Отношение приб.к уб.",
    "Сред.ежегод.чист.приб.", 
    "Наша ежегод.выручка", 
    "Отн.общей прибыли к средней прибыли", 
    "Отн.сред.выручки к сред.убытку", 
    "Отн.чистой прибыли к ср.выручке", 
    "Ср.чист.приб.клиента в год с $1,000"]
    
    cols_full_names = ['Вал.пара_глубина_шаг', 
    "Количество сделок в год", 
    "Общая выручка", 
    "Общий убыток", 
    "Общая ежегодная выручка", 
    "Общий убыток в год", 
    "Средняя выручка", 
    "Средний убыток", 
    "Отношение прибыли к убыткам",
    "Средняя ежегодная чистая прибыль", 
    "Наша ежегодная выручка", 
    "Отношение общей прибыли к средней прибыли", 
    "Отношение средней выручки к среднему убытку", 
    "Отношение чистой прибыли к средней выручке", 
    "Средняя чистая прибыль клиента в год с $1000"]
    
    cols_without_idxcol = cols[1:]
    
    cols_fullnames_without_idxcol = cols_full_names[1:]
    
    row_amnt = len(cols_without_idxcol)
    idx = list(range(1, row_amnt+1))
    # fig = make_subplots(rows=row_amnt, cols=1, specs=[[{"secondary_y": True}]])

    shoulder_50_list = os.listdir(base_dir + shoulder_50)
    shoulder_100_list = os.listdir(base_dir + shoulder_100)

    if (
        (file_name_shoulder_50 in shoulder_50_list)
        and
        (file_name_shoulder_100 in shoulder_100_list)
    ):
        df_shoulder_50 = pd.read_csv(file_dir_shoulder_50, index_col='Вал.пара_глубина_шаг', usecols=cols)
        df_shoulder_100 = pd.read_csv(file_dir_shoulder_100, index_col='Вал.пара_глубина_шаг', usecols=cols)
        list_dict = [
            [{"secondary_y": True}]
        ]

        if(save_html is True):
            if(currency_pair not in os.listdir(backtest_html_dir)):
                os.makedirs(backtest_html_dir + currency_pair)
            
        if(save_img is True):
            if(currency_pair not in os.listdir(backtest_img_dir)):
                os.makedirs(backtest_img_dir + currency_pair)
        
        for i in range(row_amnt):
            
            fig = make_subplots(specs=list_dict, rows=1,
                            cols=1)

            fig_line = go.Scatter(
                x=idx,
                y=df_shoulder_50[cols_without_idxcol[i]],
                name=f'{cols_without_idxcol[i]} 1:50',
            )

            fig.add_trace(fig_line, row=1, col=1)

            fig_line_1 = go.Scatter(
                x=idx,
                y=df_shoulder_100[cols_without_idxcol[i]],
                name=f'{cols_without_idxcol[i]} 1:100',
            )
            
            fig.add_trace(fig_line_1, row=1, col=1)


            fig.update_layout(xaxis_rangeslider_visible=False, width=1200, height=400)
            
            fig.add_annotation(
            text=f'Валютная пара: {currency_pair}, {cols_fullnames_without_idxcol[i]}', 
            xref="paper", 
            yref="paper", 
            x=0.5, 
            xanchor="center", 
            y=1.1, 
            yanchor="bottom", 
            showarrow=False) 
            
            fig.update_xaxes(title_text='№ прогона')
            fig.update_yaxes(title_text='Значение')
            
            if(save_html is True):
                if (cols_fullnames_without_idxcol[i] + '.html' not in os.listdir(backtest_html_dir + currency_pair)):
                    fig.write_html(backtest_html_dir + currency_pair + "/" + cols_fullnames_without_idxcol[i] + ".html")
                
            if(save_img is True):
                if(cols_fullnames_without_idxcol[i] + '.png' not in os.listdir(backtest_img_dir + currency_pair)):
                    file_path = str(backtest_img_dir + currency_pair + "/" + cols_fullnames_without_idxcol[i] + ".png")
                    file_name = 'png'
                    # print(file_path)
                    fig.write_image(file_path, file_name)

            fig.show()
            
            
        


    else:
        print('Ошибка, данной валютной пары не найдено.')    

def draw_multiple_currencies_pair_backtest_analytics(sav_html=False, sav_img=False) -> None:
    currency_pairs = get_currency_pairs()    
    for currency_pair in currency_pairs:
        draw_single_currency_pair_backtest_analytics(currency_pair=currency_pair, save_img=sav_img, save_html=sav_html)

def draw_longterm_graph(df: pd.DataFrame, aliquot_index:int, currency_pair:str, color='Green') -> None:
    
    df_combed = df.iloc[::aliquot_index, :]

    # Создание пустой фигуры (канваса) для графика
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_width=[0.25, 0.75])

    # Создание свечей
    
    fig_line = go.Scatter(
    x=df_combed.index,
    y=df_combed['Open'],
    name=f'График {currency_pair}',
    marker=dict(
      color=color
    )
  )

    # Создание графика объема под свечным графиком
    fig_vol = go.Bar(x=df_combed.index, y=df_combed['Volume'], name='График объема')

    # Добавление свечей, точек и объема на канвас 
    fig.add_trace(fig_line, row=1, col=1)
    fig.add_trace(fig_vol, row=2, col=1)


    # Команда нужна для того, чтобы сделать невидимым слайдер (у Plotly есть внизу слайдер, который мне показался неудобным, если необходимо его включить, надо сделать данную команду True 
    # или закомментировать всю строку, т.к. по умолчанию значение у этой команды и так True)

    fig.update_layout(xaxis_rangeslider_visible=False)


    fig.show()

def draw_multiple_longterm_graphs(df_list: list) -> None:
    
    # Создание пустой фигуры (канваса) для графика
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    for i, item in enumerate(df_list):
        df_combed = item['df'].iloc[::item['aliquot_index'], :]
        
        if (i == 0):   
            # Создание свечей    
            fig_line = go.Scatter(
            x=df_combed.index,
            y=df_combed['Open'],
            name=f'График {item["name"]}',
            marker=dict(
            color=item['color']
                )
            )     
            fig.add_trace(fig_line, row=1, col=1, secondary_y=False)
            # fig.update_yaxes(title_text=item["name"], secondary_y=False)
        else:
            fig_line = go.Scatter(
            x=df_combed.index,
            y=df_combed['Open'],
            name=f'График {item["name"]}',
            marker=dict(
            color=item['color']
                )
            )  
            fig.add_trace(fig_line, row=1, col=1, secondary_y=True)
            # 
        if(i==0):
            fig.update_yaxes(title_text=item["name"], secondary_y=False)
        else:
            fig.update_yaxes(title_text=item["name"], secondary_y=True)
        # Добавление свечей, точек и объема на канвас 

    # Команда нужна для того, чтобы сделать невидимым слайдер (у Plotly есть внизу слайдер, который мне показался неудобным, если необходимо его включить, надо сделать данную команду True 
    # или закомментировать всю строку, т.к. по умолчанию значение у этой команды и так True)

    fig.update_layout(xaxis_rangeslider_visible=False)


    fig.show()