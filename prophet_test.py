import pyupbit
import numpy as np
import pandas as pd
import datetime
import timeit
import time
import random
import sys
from json.decoder import JSONDecodeError
import requests
import json
from fbprophet import Prophet
from sklearn.linear_model import LinearRegression

df= pd.DataFrame({'name': [], 'open' : [], 'min60':[], 'min30':[], 'min15':[], 'min10':[], 'average':[], 'diff_percent':[]})

def post_message(*values):
    """슬랙 메시지 전송"""
    myToken = "xoxb-2120785924737-2120807096337-Omv2JW7ryhY6QBAuR0zJszGZ"
    texts = []
    for item in values:
        texts.append(str(item))
    str_text = ' '.join(texts)
    response = requests.post("https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer "+myToken},
        data={"channel": '#online_kyt',"text": str_text}
    )


def print_to_slack(*anyText):
    print(*anyText)
    post_message(*anyText)

def get_data(ticker, interval, count):
    # if (interval == "day"):
    param = count/200
    df_tail = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
    for i in np.arange(1, param, 1):
        date = df_tail.index[0]+datetime.timedelta(days=1)
        date_string = date.strftime("%Y%m%d")
        df_sub = pyupbit.get_ohlcv(ticker, interval=interval, to=date_string)
        df_tail = pd.concat([df_sub, df_tail]).drop_duplicates()

        # param=param-200
        print("get_ohlcv called, ", interval)
        time.sleep(0.08)

    df_final = df_tail
    print("length is ", len(df_final))

    # elif (interval == "min240"):
    #   param = count
    #   df_tail = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
    #   while param/200 > 1:
    #     date = df_tail.index[0]+datetime.timedelta(days=1)
    #     date_string = date.strftime("%Y%m%d")
    #     df_sub = pyupbit.get_ohlcv(ticker, interval=interval, to=date_string)
    #     df_tail = pd.concat([df_sub,df_tail]).drop_duplicates()

    #     param=param-200

    #   df_final = df_tail

    # check = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
    return df_final
def _ewma(arr_in, window):
    r"""Exponentialy weighted moving average specified by a decay ``window``
    to provide better adjustments for small windows via:

        y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
               (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).

    Parameters
    ----------
    arr_in : np.ndarray, float64
        A single dimenisional numpy array
    window : int64
        The decay window, or 'span'

    Returns
    -------
    np.ndarray
        The EWMA vector, same length / shape as ``arr_in``

    Examples
    --------
    >>> import pandas as pd
    >>> a = np.arange(5, dtype=float)
    >>> exp = pd.DataFrame(a).ewm(span=10, adjust=True).mean()
    >>> np.array_equal(_ewma_infinite_hist(a, 10), exp.values.ravel())
    True
    """
    n = arr_in.shape[0]
    ewma = np.empty(n)
    alpha = 1 / window
    w = 1
    ewma_old = arr_in[0]
    ewma[0] = ewma_old
    for i in range(1, n):
        w += (1-alpha)**i
        ewma_old = ewma_old*(1-alpha) + arr_in[i]
        ewma[i] = ewma_old / w
    return ewma

def rsi_ewm_np(data, window):
    array = data.to_numpy().flatten()
    chg = np.diff(array)
    gain = np.ma.masked_array(chg, mask=chg<0)
    gain = gain.filled(fill_value=0)

    loss = np.ma.masked_array(chg, mask=chg>0)
    loss = loss.filled(fill_value=0)

    avg_gain = np.append(np.zeros(window) + np.nan, _ewma(gain, window)[13:])
    avg_loss = np.append(np.zeros(window) + np.nan, _ewma(abs(loss), window)[13:])

    rs = abs(avg_gain/avg_loss)
    rsi = 100-(100/(1+rs))    
    return rsi

def predict_rsi(data, grid, unit, current_open_price):
    train_df = data.copy()
    one_coin_rsi = rsi_ewm_np(train_df.iloc[:,[3]], 14)

    train_rsi = pd.DataFrame()

    train_rsi['ds'] = train_df.iloc[14:].index
    train_rsi['y'] = one_coin_rsi[14:]
    train_rsi['o']= train_df.iloc[14:]['open'].values

    m = Prophet()
    m.add_regressor('o')
    m.fit(train_rsi)

    future = list()
    chkpnt =  train_df.iloc[14:].index[-1]
    for i in range(1, grid+1):
        date = chkpnt + datetime.timedelta(minutes=i*unit)
        future.append([date])
    future = pd.DataFrame(future)
    future.columns = ['ds']
    future['ds']= pd.to_datetime(future['ds'])

    future['o']= np.append(train_rsi['o'],[current_open_price])[-grid:]

    forecast = m.predict(future)

    # m.plot(forecast)
    # m.plot_components(forecast)

    return one_coin_rsi, forecast


def predict_rsi_only(data, grid, unit):
    train_df = data.copy()
    one_coin_rsi = rsi_ewm_np(train_df.iloc[:,[3]], 14)

    train_rsi = pd.DataFrame()

    train_rsi['ds'] = train_df.iloc[14:].index
    train_rsi['y'] = one_coin_rsi[14:]

    m = Prophet()
    m.fit(train_rsi)

    future = list()
    chkpnt =  train_df.iloc[14:].index[-1]
    for i in range(1, grid+1):
        date = chkpnt + datetime.timedelta(minutes=i*unit)
        future.append([date])
    future = pd.DataFrame(future)
    future.columns = ['ds']
    future['ds']= pd.to_datetime(future['ds'])

    forecast = m.predict(future)

    # m.plot(forecast)
    # m.plot_components(forecast)

    return one_coin_rsi, forecast

def predict_close(data, grid, unit, rsi):
    train_df = data.copy()
    one_coin_rsi = rsi_ewm_np(train_df.iloc[:,[3]], 14)

    train_rsi = pd.DataFrame()

    train_rsi['ds'] = train_df.iloc[14:].index
    train_rsi['y'] = train_df['close'][14:].values
    train_rsi['rsi'] = one_coin_rsi[14:]

    m = Prophet()
    m.add_regressor('rsi')
    m.fit(train_rsi)

    future = list()
    chkpnt =  train_df.iloc[14:].index[-1]
    for i in range(1, grid+1):
        date = chkpnt + datetime.timedelta(minutes=i*unit)
        future.append([date])
    future = pd.DataFrame(future)
    future.columns = ['ds']
    future['ds']= pd.to_datetime(future['ds'])
    future['rsi'] = rsi[-grid:]

    forecast = m.predict(future)

    # m.plot(forecast)
    # m.plot_components(forecast)

    return forecast

# def predict_rsi(data, grid, unit, current_open_price):
#     train_df = data.copy()
#     one_coin_rsi = rsi_ewm_np(train_df.iloc[:,[3]], 14)

#     train_rsi = pd.DataFrame()

#     train_rsi['ds'] = train_df.iloc[14:].index
#     train_rsi['y'] = one_coin_rsi[14:]
#     train_rsi['o']=train_df.iloc[14:]['open'].values

#     m = Prophet()
#     m.add_regressor('o')
#     m.fit(train_rsi)

#     future = m.make_future_dataframe(periods=grid, freq=unit)
#     future['o']= np.append(train_rsi['o'],[current_open_price])

#     forecast = m.predict(future)

#     # m.plot(forecast)
#     # m.plot_components(forecast)

#     return one_coin_rsi, forecast

# def predict_rsi_only(data, grid, unit):
#     train_df = data.copy()
#     one_coin_rsi = rsi_ewm_np(train_df.iloc[:,[3]], 14)

#     train_rsi = pd.DataFrame()

#     train_rsi['ds'] = train_df.iloc[14:].index
#     train_rsi['y'] = one_coin_rsi[14:]

#     m = Prophet()
#     m.fit(train_rsi)

#     future = m.make_future_dataframe(periods=grid, freq=unit)
#     forecast = m.predict(future)

#     # m.plot(forecast)
#     # m.plot_components(forecast)

#     return one_coin_rsi, forecast

# def predict_close(data, grid, unit, rsi):
#     train_df = data.copy()
#     one_coin_rsi = rsi_ewm_np(train_df.iloc[:,[3]], 14)

#     train_rsi = pd.DataFrame()

#     train_rsi['ds'] = train_df.iloc[14:].index
#     train_rsi['y'] = train_df['close'][14:].values
#     train_rsi['rsi'] = one_coin_rsi[14:]

#     m = Prophet()
#     m.add_regressor('rsi')
#     m.fit(train_rsi)

#     future = m.make_future_dataframe(periods=grid, freq=unit)
#     future['rsi']= rsi

#     forecast = m.predict(future)

#     # m.plot(forecast)
#     # m.plot_components(forecast)

#     return forecast
while True: 
    try:
        tickers = pyupbit.get_tickers(fiat="KRW")
        # tickers = ['KRW-AXS', 'KRW-ETC', 'KRW-EOS', 'KRW-XRP', 'KRW-ETH', 'KRW-BTC', 'KRW-DOGE', 'KRW-QTUM']
        total_df = {}
        tickers = pyupbit.get_tickers(fiat="KRW")

        start = datetime.datetime.now()
        t = timeit.default_timer()

        print_to_slack("starting time is ", start)
        for i in range(len(tickers)):
            one_coin = get_data(tickers[i], interval="minute60", count=200)[:-2]
            total_df[tickers[i]] = one_coin
            time.sleep(random.randint(1, 10)/17)

        end = datetime.datetime.now().time()
        t2 = timeit.default_timer() - t
        print_to_slack("ended at ", end, "difference for getting data is ", t2)


        time_checkpoint = total_df['KRW-BTC'].index[-1] + datetime.timedelta(hours=2)
        # trade_df = np.empty([1, ??? ])

        # run code in an hour unit
        while True:
            # try:
            now = datetime.datetime.now()
            print_to_slack("-------now time is ", now, "checkpoint is ", time_checkpoint)
            print_to_slack(" ")
            print_to_slack(" ")

            # do this job evey 1 hour
            if time_checkpoint <= now < time_checkpoint + datetime.timedelta(hours=1) - datetime.timedelta(seconds=10):

                # update total_df by adding data from the last hour
                print_to_slack("-------updating data--------------")
                tickers = pyupbit.get_tickers(fiat="KRW")
                filtered_by_key = {key: total_df[key]['volume'][-1]*total_df[key]['close'][-1] for key in tickers}
                tickers = [a_tuple[0] for a_tuple in sorted(filtered_by_key.items(), key=lambda x: x[1])]  #filter from lower volume to high, in order to prevent update loss (although we check timecheckpoint)

                # print_to_slack(tickers)
                start = datetime.datetime.now()
                t = timeit.default_timer()

                print_to_slack("starting time is ", start)
                for i in range(len(tickers)):
                    # isit_uptodate = time_checkpoint
                    # while isit_uptodate != (time_checkpoint - datetime.timedelta(hours=1)):            #initial value would be 1 hour ago beacuse when it was called last was before loop
                    #     one_coin_data = pd.DataFrame(pyupbit.get_ohlcv(tickers[i], interval="minute60", count=2))
                    #     print_to_slack(one_coin_data)
                    #     one_coin = one_coin_data.iloc[0]
                    #     isit_uptodate = one_coin.name
                    one_coin_data = pyupbit.get_ohlcv(tickers[i], interval="minute60", count=2)   #due to delayed update, this might be either [now-2,now-1] or [now-1, now] so we need to filter

                    if one_coin_data.index[-1] == (time_checkpoint - datetime.timedelta(hours=1)):   #if [now-2, now-1]
                        one_coin = one_coin_data.iloc[1]
                        total_df[tickers[i]] = total_df[tickers[i]].iloc[1:].append(one_coin).drop_duplicates()
                        # if tickers[i] in ["KRW-BTC","KRW-MTL","KRW-NEO","KRW-IOST","KRW-EFL", "KRW-ETH", "KRW-ARK"]:
                        if tickers[i] in ["KRW-BTC", "KRW-ETH"]:
                            print_to_slack(total_df[tickers[i]].iloc[[0,1,-2,-1]])
                            print_to_slack(" ")
                            print_to_slack(pd.DataFrame(one_coin).T)                

                    elif one_coin_data.index[-1] == time_checkpoint:   #if [now-1, now]
                        one_coin = one_coin_data.iloc[0]
                        total_df[tickers[i]] = total_df[tickers[i]].iloc[1:].append(one_coin).drop_duplicates()
                        # if tickers[i] in ["KRW-BTC","KRW-MTL","KRW-NEO","KRW-IOST","KRW-EFL", "KRW-ETH", "KRW-ARK"]:
                        if tickers[i] in ["KRW-BTC","KRW-ETH"]:
                            print_to_slack(total_df[tickers[i]].iloc[[0,1,-2,-1]])
                            print_to_slack(" ")
                            print_to_slack(pd.DataFrame(one_coin).T)

                    else:
                        print_to_slack("something's wrong with updating")                  

                    time.sleep(random.randint(1, 10)/30)


                end = datetime.datetime.now().time()
                print_to_slack({key: len(total_df[key]['volume']) for key in tickers})
                print_to_slack("ended at ", end, "difference for getting data is ",
                    timeit.default_timer() - t)

                #-----------------------------------------------------------------------------------
                print_to_slack("-------getting 4 high rsi candidates (tail minimum1 is 2)--------------")

                # get high rsi candidates
                coins_rsi = {}
                coin_number = 0
                high_rsi_cutoff = 60
                tail_minimum1 = 3       # minimum back testing time frame is 24 hours
                # total_backup = total_df.copy()
                high_rsi_increasing = []
                high_increasing_trend = []
                high_increasing_trend_2 = []
                slopes = []

                while high_rsi_cutoff >= 55:
                    while tail_minimum1 <=10:
                            
                        for coin in total_df:
                            one_coin_rsi = rsi_ewm_np(total_df[coin].iloc[:,[3]], 14)
                            coins_rsi[coin] = np.around(one_coin_rsi[-tail_minimum1:].mean(), 3)

                            x = np.arange(0, tail_minimum1, 1).reshape((-1, 1))
                            y = one_coin_rsi[-tail_minimum1:]

                            model = LinearRegression().fit(x, y)
                            slope = float(model.coef_)

                            if (one_coin_rsi[-3]<one_coin_rsi[-2]) & (one_coin_rsi[-2]<one_coin_rsi[-1]) & ((one_coin_rsi[-1]-one_coin_rsi[-3]) > 20):
                                high_rsi_increasing.append(coin)
                                if slope > 4.5:
                                    high_increasing_trend_2.append(coin)
                                    slopes.append(slope)
                                elif slope > 1: 
                                    high_increasing_trend.append(coin)

                            else:
                                if slope > 4.5:
                                    high_increasing_trend_2.append(coin)    
                                    slopes.append(slope)     
                                elif slope > 1: 
                                    high_increasing_trend.append(coin)                                           
                            # print_to_slack(tail_minimum1, "got high rsi for", coin)

                        # high_rsi = {key: value for (key, value) in coins_rsi.items() if (coins_rsi[key] >= high_rsi_cutoff) & (total_df[key]['volume'][-1] >= total_df[key]['volume'][-2])}
                        high_rsi = {key: value for (key, value) in coins_rsi.items() if (coins_rsi[key] >= high_rsi_cutoff) & (key in (high_increasing_trend+high_increasing_trend_2))}
                        high_rsi_2 = {key: value for (key, value) in coins_rsi.items() if (key in high_increasing_trend_2)}
                        
                        coin_number = len(high_rsi)
                        if coin_number > 0:
                            break
                        else:
                            print_to_slack(tail_minimum1, "got high rsi for all coins with high rsi cutoff", high_rsi_cutoff)
                            tail_minimum1 += 1  

                    high_rsi_cutoff -= 0.5
                    tail_minimum1 =3
                    if coin_number > 0:
                        break
                    

                candidates_high = [*high_rsi]
                candidates_high_2 = [*high_rsi_2]
                print_to_slack("before filtering by capacity, candidates_high are ", candidates_high)
                print_to_slack("before filtering by capacity, candidates_high_2 are ", candidates_high_2)
                print_to_slack("slopes", slopes)
                print(coins_rsi)

                max_number = 4 ######## maximum number of coins to bid, sort by rsi index 

                if coin_number > max_number:  # if there are too may candidates, filter them to 6 biggest rsi
                    # candidates_high = [a_tuple[0] for a_tuple in sorted(
                    #     high_rsi.items(), key=lambda x: x[1], reverse=True)[:max_number]]
                    filtered_by_key = {key: total_df[key]['volume']*total_df[key]['close'] for key in candidates_high}
                    candidates_high = [a_tuple[0] for a_tuple in sorted(filtered_by_key.items(), key=lambda x: x[1].tail(14+tail_minimum1).mean(), reverse=True)[:max_number]]  #filter high rsi candidates by volume size
                    # coin_number = 6

                if len(high_rsi_2) > max_number:
                    filtered_by_key_2 = {key: total_df[key]['volume']*total_df[key]['close'] for key in [*high_rsi_2]}
                    candidates_high_2 = [a_tuple[0] for a_tuple in sorted(filtered_by_key_2.items(), key=lambda x: x[1].tail(14+tail_minimum1).mean(), reverse=True)[:max_number]]            
                
                print_to_slack("high rsi candidates are ", candidates_high)
                print_to_slack("high rsi_2 candidates are ", candidates_high_2)

                print_to_slack({key:total_df[key].iloc[[0,-2,-1]] for key in candidates_high})
                #------------total
                #-----------------------------------------------------------------------------------
                print_to_slack("-------getting 4 low rsi(with high volume, tail_minimum2=30) candidates--------------")

                # get high rsi candidates
                coins_rsi = {}
                coin_number = 0
                low_rsi_cutoff = 45
                tail_minimum2 = 30
                # total_backup = total_df.copy()
                rsi_m_appearance = []

                while low_rsi_cutoff <= 50:
                    while tail_minimum2 <=140:
                            
                        for coin in total_df:
                            one_coin_rsi = rsi_ewm_np(total_df[coin].iloc[:,[3]], 14)
                            coins_rsi[coin] = np.around(one_coin_rsi[-tail_minimum2:].mean(), 3)
                            if (one_coin_rsi[-3]>one_coin_rsi[-2]) & (one_coin_rsi[-2]<one_coin_rsi[-1]):
                                rsi_m_appearance.append(coin)
                            
                            # print_to_slack(tail_minimum1, "got high rsi for", coin)

                        # low_rsi = {key: value for (key, value) in coins_rsi.items() if (coins_rsi[key] < low_rsi_cutoff)  & (total_df[key]['volume'][-1] >= total_df[key]['volume'][-2])}
                        low_rsi = {key: value for (key, value) in coins_rsi.items() if (coins_rsi[key] < low_rsi_cutoff) }
                        coin_number = len(low_rsi)

                        if coin_number > 0:
                            break
                        else:
                            if tail_minimum2 % 20 == 0:
                                print_to_slack(tail_minimum2, "got low rsi for all coins with low rsi cutoff ", low_rsi_cutoff)
                            tail_minimum2 += 1  

                    low_rsi_cutoff += 1
                    tail_minimum2 = 30
                    if coin_number > 0:
                        break



                candidates_low = [x for x in [*low_rsi] if (not x in candidates_high) ]
                # rsi_m_appearance = [x for x in rsi_m_appearance if not x in candidates_high]  # no probability that this happens
                
                print_to_slack("before filtering by capacity, candidates_low are ", candidates_low)
                print(coins_rsi)

                coin_number = len(candidates_low)

                max_number = 8 - len(candidates_high)  ######## maximum number of coins to bid, sort by rsi index 

                if coin_number > max_number:  # if there are too may candidates, filter them to 6 biggest rsi
                    filtered_by_key = {key: total_df[key]['volume']*total_df[key]['close'] for key in candidates_low}
                    candidates_low = [a_tuple[0] for a_tuple in sorted(filtered_by_key.items(), key=lambda x: x[1].tail(14+tail_minimum2).median(), reverse=True)[:max_number]] 


                    # coin_number = 6
                print_to_slack("low rsi candidates are ", candidates_low, "cutoff is ", low_rsi_cutoff)
                print_to_slack({key:total_df[key].iloc[[0,-2,-1]] for key in candidates_low})

                #------------------------------------------------------------------------------------



                opening_prices = []
                candidates = candidates_high +  candidates_low

                print_to_slack("-------candidates are", candidates)
                coin_number = len(candidates)

                candidates = ["KRW-XLM"]
                while now < time_checkpoint + datetime.timedelta(hours=1) - datetime.timedelta(seconds=10):
                    for coin in candidates:
                        now = datetime.datetime.now()
                        print_to_slack(now)
                        minute = now.minute

                        grid30 = 2-int(minute/30)
                        grid15 = 4-int(minute/15)
                        grid10 = 6-int(minute/10)    
                        
                        data_min60_600 = get_data(coin, interval = "minute60", count=500)
                        data_min30_1200 =  get_data(coin, interval = "minute30", count=1000)
                        data_min15_3000 =  get_data(coin, interval = "minute15", count=2000)
                        data_min10_3000 =  get_data(coin, interval = "minute10", count=2500)

                        hours_before = 1

                        rsi60, rsi60_pred = predict_rsi(data=data_min60_600.iloc[:-hours_before], grid=1, unit=60, current_open_price=data_min60_600['open'][-hours_before])
                        rsi30, rsi30_pred = predict_rsi_only(data=data_min30_1200.iloc[:-(hours_before*2-1)], grid=grid30, unit=30)
                        rsi15, rsi15_pred = predict_rsi_only(data=data_min15_3000.iloc[:-(hours_before*4-3)], grid=grid15, unit=15)
                        rsi10, rsi10_pred = predict_rsi_only(data=data_min10_3000.iloc[:-(hours_before*6-5)], grid=grid10, unit=10)


                        close_min60 = predict_close(data=data_min60_600.iloc[:-hours_before], grid=1, unit=60, rsi=np.append(rsi60[14:],[rsi60_pred['yhat'].values[-1]]))
                        close_min30 = predict_close(data=data_min30_1200.iloc[:-(hours_before*2-1)], grid=grid30, unit=30, rsi=np.append(rsi30[14:],[rsi30_pred['yhat'].values[-grid30:]]))
                        close_min15 = predict_close(data=data_min15_3000.iloc[:-(hours_before*4-3)], grid=grid15, unit=15, rsi=np.append(rsi15[14:],[rsi15_pred['yhat'].values[-grid15:]]))
                        close_min10 = predict_close(data=data_min10_3000.iloc[:-(hours_before*6-5)], grid=grid10, unit=10, rsi=np.append(rsi10[14:],[rsi10_pred['yhat'].values[-grid10:]]))
                        print_to_slack(close_min10['ds'].values[-1], close_min15['ds'].values[-1],close_min30['ds'].values[-1],close_min60['ds'].values[-1])

                        # (close_min60['yhat'].values[-1]+ close_min30['yhat'].values[-1]+ close_min15['yhat'].values[-1]+ close_min10['yhat'].values[-1])/4
                        # one_line= pd.DataFrame({'name': [], 'open' : [], 'min60':[], 'min30':[], 'min15':[], 'min10':[], 'average':[], 'diff_percent':[]})
                        average_price = (close_min60['yhat'].values[-1]+close_min30['yhat'].values[-1]+close_min15['yhat'].values[-1]+close_min10['yhat'].values[-1])/4
                        df = df.append({'name': coin, 'open' : data_min60_600['open'][-hours_before], 'min60':close_min60['yhat'].values[-1], 'min30':close_min30['yhat'].values[-1], 'min15':close_min15['yhat'].values[-1], 'min10':close_min10['yhat'].values[-1], 'average':(close_min60['yhat'].values[-1]+close_min30['yhat'].values[-1]+close_min15['yhat'].values[-1]+close_min10['yhat'].values[-1])/4, 'diff_percent':(average_price - data_min60_600['open'][-hours_before])*100/data_min60_600['open'][-hours_before]},ignore_index=True)

                        print_to_slack(coin, "open: ", data_min60_600['open'][-hours_before], close_min60['yhat'].values[-1], close_min30['yhat'].values[-1], close_min15['yhat'].values[-1], close_min10['yhat'].values[-1], "---diff: ", (average_price - data_min60_600['open'][-hours_before])*100/data_min60_600['open'][-hours_before])
                        # one_line['name'].append(coin)
                        # one_line['open'].append(data_min60_600['open'][-hours_before])
                        # one_line['min60'].append(close_min60['yhat'].values[-1])
                        # one_line['min30'].append(close_min30['yhat'].values[-1])
                        # one_line['min15'].append(close_min15['yhat'].values[-1])
                        # one_line['min10'].append(close_min10['yhat'].values[-1])
                        # one_line['average'].append((close_min60['yhat'].values[-1]+close_min30['yhat'].values[-1]+close_min15['yhat'].values[-1]+close_min10['yhat'].values[-1])/4)
                        # one_line['diff_percent'].append((one_line['open'] - one_line['average'])*100/one_line['open'])
                        # df = pd.concat([df, one_line])
                        now = datetime.datetime.now()
                        if now >= time_checkpoint + datetime.timedelta(hours=1) - datetime.timedelta(seconds=10):
                            break

            elif now < time_checkpoint + datetime.timedelta(hours=1):
                while now < time_checkpoint + datetime.timedelta(hours=1):
                    wating_final = time_checkpoint + datetime.timedelta(hours=1) - now 
                    print_to_slack("---all coins are sold, wating for next loop: ", wating_final)
                    time.sleep((wating_final).total_seconds())
                    now = datetime.datetime.now()

                time_checkpoint = time_checkpoint + datetime.timedelta(hours=1)

            else:
                print_to_slack("SOMETHING might be WRONG if this msg repeats!!!")
                time.sleep(1)        
                now = datetime.datetime.now()
                time_checkpoint = time_checkpoint + datetime.timedelta(hours=1)
  
    except Exception as e:
        print (e)
        print ('Restarting!')
        time.sleep(2)
        continue
    else:
        break
