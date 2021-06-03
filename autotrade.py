import pyupbit
import pandas as pd
import numpy as np
import os
import datetime
import time
import random
import timeit
import requests
from sklearn.linear_model import LinearRegression
# import logging 
from fbprophet import Prophet
import sys
# old_stdout = sys.stdout

# log_file = open("output.log","w")
# sys.stdout = log_file

# sys.stdout = old_stdout
# log_file.close()

# myToken = "xoxb-2120785924737-2120807096337-Omv2JW7ryhY6QBAuR0zJszGZ"

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

    check = 1
    while check > 0:
        try:
            post_message(*anyText)
            check -= 1

        except Exception as e:
            print (e)
            print ('Restarting print_to_slack!')
            time.sleep(2)
            continue


def rsi_ewm(data, period):         #input is daaframe, output is numpy
    delta = data.diff()

    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    _gain = up.ewm(com=(period - 1), min_periods=period).mean()
    _loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()

    RS = _gain / _loss
    return (100 - (100 / (1 + RS))).to_numpy().flatten()


def np_rsi_plain(array, cutoff):
    rsi_period = cutoff
    chg = np.diff(array[:, 3])
    gain = np.ma.masked_array(chg, mask=chg < 0)
    gain = gain.filled(fill_value=0)

    loss = np.ma.masked_array(chg, mask=chg > 0)
    loss = loss.filled(fill_value=0)

    avg_gain = np.append(np.zeros(cutoff) + np.nan,
                         moving_average(gain, cutoff))
    avg_loss = np.append(np.zeros(cutoff) + np.nan,
                         moving_average(loss, cutoff))

    rs = abs(avg_gain/avg_loss)
    rsi = 100-(100/(1+rs))

    return rsi

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


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def np_draw_frame_ma_per_mean(data, k, per, fee, ma, splits, high_index):
    data_pd = data.copy()
    df = data_pd.to_numpy()

    ma5 = np.append(np.zeros(5) + np.nan, moving_average(df[:, 3], 5)[:-1])
    # df['ma5'] = df['close'].rolling(window=5).mean().shift(1)

    # 변동폭 계산
    if high_index:
        range_coin = abs(df[:, 0]-df[:, 3])  # if high rsi, more aggressively inveset
        rsi_numpy = np.append(np.zeros(1) + np.nan, rsi_ewm_np(data_pd.iloc[:,[3]], 14)[:-1])

    elif high_index ==0:
        range_coin = df[:, 1]-df[:, 2]       # if not high rsi, more conservatively invest
        rsi_numpy = np.zeros(len(df)) + 60  # so rsi_numpy >50 can be always true in low_rsi

    # df['range'] = (df['high'] - df['low'])

    # 당일 기준 splits 일 전까지의 변동폭들의 mean*k를 open에 더해서 target으로 잡는다.
    # check = [np.nan]*(splits-1)
    # range_averages = [sum(df['range'][i:i + splits])/splits for i in range(len(df['range']) - splits + 1)]
    # check.extend(range_averages)
    # df['range_average'] = check
    range_average = np.append(
        np.zeros(splits) + np.nan, moving_average(range_coin, splits)[:-1])
    # df['range_average'] = df['range'].rolling(window=splits).mean()

    # target(매수가), range_averages 컬럼을 한칸씩 밑으로 내림(.shift(1))
    target = df[:, 0] + range_average*k
    close_to_target = (df[:, 3]/target - 1) * 100
    close_to_open = (df[:, 3]/df[:, 0] - 1) * 100

    # df['target'] = df['open'] + df['range_average'].shift(1)*k
    # df['close_to_target'] = (df['close']/df['target']-1)*100
    # df['close_to_open'] = (df['close']/df['open']-1)*100

    # create rsi
    # rsi_numpy = np.append(np.zeros(1) + np.nan, np_rsi_v2(df, 14)[:-1])
    # df['rsi'] = rsi(df,14).shift(1)     #shifted by 1 unit because it is what we refer to when considering at current timepoint

    if ma:
        # df['bull'] = df['open'] > df['ma5']
        # ror(수익률), np.where(조건문, 참일때 값, 거짓일때 값), 사고팔 때 1-fee/100만큼 곱해야 하니까 두번 곱한다.
        ror = np.where((df[:, 1] >= target) & (df[:, 1] >= (1+per/100)*target) & (df[:, 0] >= ma5) & (rsi_numpy >= 50),
                       (1+per/100)*(1-fee/100)*(1-fee/100),
                       np.where((df[:, 1] >= target) & (df[:, 1] < (1+per/100)*target) & (df[:, 0] >= ma5) & (rsi_numpy >= 50),
                                (df[:, 3] / target) *
                                (1-fee/100)*(1-fee/100),
                                1))

        # df['ror'] = np.where((df['high'] > df['target']) & (df['high']>= (1+per/100)*df['target']) & (df['open'] > df['ma5']) & (df['rsi']>=50),
        #                     (1+per/100)*(1-fee/100)*(1-fee/100),
        #                     np.where((df['high'] > df['target']) & (df['high']< (1+per/100)*df['target']) & (df['open'] > df['ma5']) & (df['rsi']>=50),
        #                               (df['close'] / df['target'])*(1-fee/100)*(1-fee/100),
        #                               1) )

        # df['ror'] = np.where((df['high'] > df['target']) & (df['high']< (1+per/100)*df['target']) & df['bull'],
        #                     (df['close'] / df['target'])*(1-fee/100)*(1-fee/100),
        #                      1)
    elif not(ma):
        ror = np.where((df[:, 1] >= target) & (df[:, 1] >= (1+per/100)*target) & (rsi_numpy >= 50),
                       (1+per/100)*(1-fee/100)*(1-fee/100),
                       np.where((df[:, 1] >= target) & (df[:, 1] < (1+per/100)*target) & (rsi_numpy >= 50),
                                (df[:, 3] / target) *
                                (1-fee/100)*(1-fee/100),
                                1))
        # df['ror'] = np.where((df['high'] > df['target']) & (df['high']>= (1+per/100)*df['target']) & (df['rsi']>=50),
        #                 (1+per/100)*(1-fee/100)*(1-fee/100),
        #                 np.where((df['high'] > df['target']) & (df['high']< (1+per/100)*df['target']) & (df['rsi']>=50),
        #                           (df['close'] / df['target'])*(1-fee/100)*(1-fee/100),
        #                           1) )
    # 누적 곱 계산(cumprod) => 누적 수익률
    hpr = np.cumprod(ror)
    # df['hpr'] = df['ror'].cumprod()

    # Draw Down 계산 (누적 최대 값과 현재 hpr 차이 / 누적 최대값 * 100)
    dd = 100 * (np.maximum.accumulate(hpr) - hpr) / np.maximum.accumulate(hpr)
    # df['dd'] = (df['hpr'].cummax() - df['hpr']) / df['hpr'].cummax() * 100

    # MDD 계산
    # print("MDD(%): ", df['dd'].max())
    # print("HPR(%): ", df['hpr'][-1])

    new_range_average = moving_average(range_coin, splits)[-1]
    new_ma5 = moving_average(df[:, 3], 5)[-1]
    # new_rsi = rsi_ewm(data_pd.iloc[:,[3]], 14)[-1]

    return hpr[-1], dd[-1], np.sum(ror < 1), new_range_average, new_ma5

# def np_draw_frame_ma_per_mean_norsi(data, k, per, fee, ma, splits):
#     data_pd = data.copy()
#     df = data_pd.to_numpy()

#     ma5 = np.append(np.zeros(5) + np.nan, moving_average(df[:, 3], 5)[:-1])

#     # range_coin = df[:, 1]-df[:, 2]
#     range_coin = abs(df[:, 0]-df[:, 3])


#     range_average = np.append(
#         np.zeros(splits) + np.nan, moving_average(range_coin, splits)[:-1])
#     # df['range_average'] = df['range'].rolling(window=splits).mean()

#     # target(매수가), range_averages 컬럼을 한칸씩 밑으로 내림(.shift(1))
#     target = df[:, 0] + range_average*k
#     close_to_target = (df[:, 3]/target - 1) * 100
#     close_to_open = (df[:, 3]/df[:, 0] - 1) * 100

#     # df['target'] = df['open'] + df['range_average'].shift(1)*k
#     # df['close_to_target'] = (df['close']/df['target']-1)*100
#     # df['close_to_open'] = (df['close']/df['open']-1)*100

#     # create rsi
#     rsi_numpy = np.append(np.zeros(1) + np.nan, rsi_ewm(data_pd.iloc[:,[3]], 14)[:-1])


#     if ma:
#         # df['bull'] = df['open'] > df['ma5']
#         # ror(수익률), np.where(조건문, 참일때 값, 거짓일때 값), 사고팔 때 1-fee/100만큼 곱해야 하니까 두번 곱한다.
#         ror = np.where((df[:, 1] >= target) & (df[:, 1] >= (1+per/100)*target) & (df[:, 0] >= ma5) ,
#                        (1+per/100)*(1-fee/100)*(1-fee/100),
#                        np.where((df[:, 1] >= target) & (df[:, 1] < (1+per/100)*target) & (df[:, 0] >= ma5),
#                                 (df[:, 3] / target) *
#                                 (1-fee/100)*(1-fee/100),
#                                 1))

#     elif not(ma):
#         ror = np.where((df[:, 1] >= target) & (df[:, 1] >= (1+per/100)*target) ,
#                        (1+per/100)*(1-fee/100)*(1-fee/100),
#                        np.where((df[:, 1] >= target) & (df[:, 1] < (1+per/100)*target) ,
#                                 (df[:, 3] / target) *
#                                 (1-fee/100)*(1-fee/100),
#                                 1))

#     hpr = np.cumprod(ror)
#     dd = 100 * (np.maximum.accumulate(hpr) - hpr) / np.maximum.accumulate(hpr)

#     new_range_average = moving_average(range_coin, splits)[-1]
#     new_ma5 = moving_average(df[:, 3], 5)[-1]
#     new_rsi = rsi_ewm(data_pd.iloc[:,[3]], 14)[-1]

#     return hpr[-1], dd[-1], np.sum(ror < 1), new_range_average, new_ma5, new_rsi



def get_data(ticker, interval, count):
    # if (interval == "day"):
    param = count/200
    check = 1
    while check > 0:
        try:
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


            check -= 1
        except Exception as e:
            print (e)
            print ('Restarting get_data!')
            time.sleep(2)
            continue
        

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


def find_maximum_index(data, per, days, splits, back_length, high_index, splits_min):
    df = data.copy()
    # compare_data = pd.DataFrame(columns = [0,1,2,3,4,5,6,7,8])
    compare_data = []

    # per_limit = per+1
    # days_limit = days+1
    # splits_limit = splits+1

    for days in np.arange(1, days+1, 1):
        using_data = pd.DataFrame(df.iloc[::days, :][-back_length:])

        length = len(using_data)
        for per in np.arange(0.9, per+1, 0.2):
            for splits in np.arange(splits_min, splits+1, 1):
                for ma in [True, False]:

                    k_list = []
                    hpr_list = []
                    count_loss = []
                    MDD = []
                    vol_index = []
                    # per = 9
                    # ma = False
                    # splits = 1
                    sub_df = []

                    for k in np.arange(0.2, 1.1, 0.1):
                        hpr, dd, num_ror_below1, new_range_average, new_ma5 = np_draw_frame_ma_per_mean(
                            using_data, k=k, per=per, fee=0.05, ma=ma, splits=splits, high_index=high_index)
                        # elif high_index==0:
                        #     hpr, dd, num_ror_below1, new_range_average, new_ma5, new_rsi = np_draw_frame_ma_per_mean_norsi(
                        #         using_data, k=k, per=per, fee=0.05, ma=ma, splits=splits)            
                        # hpr = calculated['hpr'][-1]
                        # dd = np.max(calculated['dd'])
                        # num_ror_below1 = np.sum(calculated['ror']<1)

                        if hpr > 1:
                            # k_list.append(k)
                            # hpr_list.append(hpr)
                            # count_loss.append(num_ror_below1/len(using_data))       #num_ror_below1 is divided by the length of using_data, in a way of normalization
                            # MDD.append(dd)
                            # vol_index.append(100*hpr**3/max(0.05*5,(num_ror_below1/len(using_data))*dd)**0.9)
                            sub_df.append([days, back_length, per, splits, ma, k, hpr, num_ror_below1/length, dd, 100*hpr**3/max(
                                0.05*5, (num_ror_below1/length)*dd)**0.9, new_range_average, new_ma5])

                    # length = len(k_list)
                    # sub_df = pd.DataFrame(list(zip([days]*length, [per]*length, [splits]*length, [ma]*length, k_list, hpr_list, count_loss, MDD, vol_index)))
                    compare_data.extend(sub_df)

    # final_df = pd.DataFrame(compare_data, columns = ['days', 'per','splits','ma','k','hpr','count_loss','mdd', 'vol_index'])
    # compare_data.columns = ['days', 'per','splits','ma','k','hpr','count_loss','mdd', 'vol_index']

    return compare_data

def get_recent_tick5(coin):
    check = 1
    while check > 0:
        try:
            req    = requests.get(f'https://crix-api-endpoint.upbit.com/v1/crix/candles/ticks/1?code=CRIX.UPBIT.{coin}&count=5')
            data   = req.json()
            result = []

            for i, candle in enumerate(data):
                result.append({
                    'Time'                 : data[i]["candleDateTimeKst"], 
                    'OpeningPrice'         : data[i]["openingPrice"],
                    'HighPrice'            : data[i]["highPrice"],
                    'LowPrice'             : data[i]["lowPrice"],
                    'TradePrice'           : data[i]["tradePrice"],
                    'CandleAccTradeVolume' : data[i]["candleAccTradeVolume"],
                    "candleAccTradePrice"  : data[i]["candleAccTradePrice"]
                })

            
            check -= 1
        except Exception as e:
            print (e)
            print ('Restarting get_recent_ticks!')
            time.sleep(2)
            continue

    return result


###############################################
##############  final file ####################
access = "dihtIGpo6j9PkhrsxRGRWPTRKrkw13w8fxVze15C"
secret = "a2YBnwtTvJ6UkmyDgUCnJxA9rWqqtDUQcyuluBx1"
upbit = pyupbit.Upbit(access, secret)
fee = 0.05


# define total_df = very initial values (will be updated every 1 hour)

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
        # print_to_slack({key: len(total_df[key]['volume']) for key in tickers})
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

                    if (one_coin_rsi[-3]<=one_coin_rsi[-2]) & (one_coin_rsi[-2]<=one_coin_rsi[-1]) & ((one_coin_rsi[-1]-one_coin_rsi[-3]) > 10):
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
                # high_rsi = {key: value for (key, value) in coins_rsi.items() if (coins_rsi[key] >= high_rsi_cutoff) & (key in high_increasing_trend)}
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



        candidates_low = [x for x in [*low_rsi] if not x in candidates_high]
        # rsi_m_appearance = [x for x in rsi_m_appearance if not x in candidates_high]  # no probability that this happens
        
        print_to_slack("before filtering by capacity, candidates_low are ", candidates_low)
        print_to_slack(coins_rsi)

        coin_number = len(candidates_low)

        max_number = 8 - len(candidates_high) ######## maximum number of coins to bid, sort by rsi index 

        if coin_number > max_number:  # if there are too may candidates, filter them to 6 biggest rsi
            filtered_by_key = {key: total_df[key]['volume']*total_df[key]['close'] for key in candidates_low}
            candidates_low = [a_tuple[0] for a_tuple in sorted(filtered_by_key.items(), key=lambda x: x[1].tail(14+tail_minimum2).median(), reverse=True)[:max_number]] 


            # coin_number = 6
        print_to_slack("low rsi candidates are ", candidates_low, "cutoff is ", low_rsi_cutoff)
        print_to_slack({key:total_df[key].iloc[[0,-2,-1]] for key in candidates_low})

        #------------------------------------------------------------------------------------



        opening_prices = []
        candidates = candidates_high + candidates_low

        print_to_slack("-------candidates are", candidates)
        coin_number = len(candidates)
        # total_backup = total_df.copy()

        # get optimal values of days, per, splits, ma, k     and opening prices
        trade_df = np.empty([1, 12])
        for coin_num in range(coin_number):
            coin = candidates[coin_num]
            # optimum_check columns: ['days','back_length', 'per','splits','ma','k','hpr','count_loss','mdd', 'vol_index', new_range_average, new_ma5]
            maximums = find_maximum_index(
                total_df[coin], per=10, days=1, splits=1 if coin in rsi_m_appearance else 8, back_length=14+tail_minimum1 if (coin in candidates_high) else 14+tail_minimum2, high_index=1 if (coin in candidates_high) else 0, splits_min=4 if coin in high_rsi_increasing else 1) ##### 100 or 14+tail_minimum?
            if maximums:
                optimum_check = np.vstack(maximums)
                # there might be different optimums that have maximum index, so just pick the first one
                optimum_loc = np.where(
                    optimum_check[:, 9] == np.max(optimum_check[:, 9]))[0][0]
                optimum = optimum_check[optimum_loc]
                trade_df = np.vstack((trade_df, optimum))
            elif not maximums:  # backup just in case when maximums are empty, then increase tail_minimum by 1 to find until maximums are not empty
                new_tail = 14+tail_minimum1 if (coin in candidates_high) else 14+tail_minimum2
                while not maximums:
                    new_tail += 1
                    maximums = find_maximum_index(
                        total_df[coin], per=10, days=1, splits=1 if coin in rsi_m_appearance else 8, back_length=new_tail, high_index=1 if (coin in candidates_high) else 0, splits_min=4 if coin in high_rsi_increasing else 1)
                optimum_check = np.vstack(maximums)
                # there might be different optimums that have maximum index, so just pick the first one
                optimum_loc = np.where(
                    optimum_check[:, 9] == np.max(optimum_check[:, 9]))[0][0]
                optimum = optimum_check[optimum_loc]
                trade_df = np.vstack((trade_df, optimum))

            print_to_slack("backtesting ended for", coin)
            opening_prices.append(float(pyupbit.get_ohlcv(
                coin, interval="minute60", count=1)['open'].values))

        # take away the first array that was only there for initialization
        trade_df = trade_df[1:]
        print_to_slack("-------trade df is ",trade_df)

        ###### optimizaion (back testing) is all prepared for seeking which ones to buy in this unit of 1 hour #####
        bought = np.zeros(coin_number)
        already_sell = np.zeros(coin_number)
        too_low = np.zeros(coin_number)
        min10_data = {}
        # min10_checkpoint = np.zeros(coin_number)
        too_low_backtests = np.zeros(shape=(coin_number, 12))
        openings_min10 = np.zeros(coin_number)
        target_prices_min10 = np.zeros(coin_number)

        balances = upbit.get_balances()
        balance_kr = float(list(filter(lambda d: d['currency'] in ['KRW'], balances))[0]['balance'])/2 - 50000
        # current_prices = []

        target_prices = np.zeros(coin_number)
        current_balances = np.zeros(coin_number)
        actual_bid_prices = np.zeros(coin_number)

        unlock = np.zeros(coin_number)

        if coin_number > 0 :
            print_to_slack("starting prediction")
            t = timeit.default_timer()

            for coin_num in range(coin_number):
                coin = candidates[coin_num]

                now = datetime.datetime.now()
                print_to_slack(now)

                if now >= time_checkpoint + datetime.timedelta(hours=1) - datetime.timedelta(seconds=10):
                    break

                else:                
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
                    min10_pred = close_min10['yhat'].values[-1]
                    min15_pred = close_min15['yhat'].values[-1]
                    min30_pred = close_min30['yhat'].values[-1]
                    min60_pred = close_min60['yhat'].values[-1]

                    average_price = (close_min60['yhat'].values[-1]+close_min30['yhat'].values[-1]+close_min15['yhat'].values[-1]+close_min10['yhat'].values[-1])/4
                    # df = df.append({'name': coin, 'open' : data_min60_600['open'][-hours_before], 'min60':close_min60['yhat'].values[-1], 'min30':close_min30['yhat'].values[-1], 'min15':close_min15['yhat'].values[-1], 'min10':close_min10['yhat'].values[-1], 'average':(close_min60['yhat'].values[-1]+close_min30['yhat'].values[-1]+close_min15['yhat'].values[-1]+close_min10['yhat'].values[-1])/4, 'diff_percent':(average_price - data_min60_600['open'][-hours_before])*100/data_min60_600['open'][-hours_before]},ignore_index=True)

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

                    if  min10_pred == min(min10_pred, min15_pred, min30_pred, min60_pred):
                        larger_60 = close_min60['yhat'].values[-1] > opening_prices[coin_num]
                        larger_30 = close_min30['yhat'].values[-1] > opening_prices[coin_num]
                        larger_15 = close_min15['yhat'].values[-1] > opening_prices[coin_num]
                        larger_10 = close_min10['yhat'].values[-1] > opening_prices[coin_num]
                        
                        if (sum([larger_60, larger_30 ,  larger_15 ,  larger_10]) >=3) & (min60_pred > opening_prices[coin_num]):
                            unlock[coin_num] = 1
                    
                    else:
                        larger_60 = close_min60['yhat'].values[-1] > opening_prices[coin_num]
                        larger_30 = close_min30['yhat'].values[-1] > opening_prices[coin_num]
                        larger_15 = close_min15['yhat'].values[-1] > opening_prices[coin_num]
                        larger_10 = close_min10['yhat'].values[-1] > opening_prices[coin_num]

                        if (sum([larger_60 , larger_30, larger_15 ,  larger_10]) >=3) & (min10_pred > opening_prices[coin_num]):
                            unlock[coin_num] = 1
            t2 = timeit.default_timer() - t       
            print_to_slack("PREDICTION TIME is ", t2)
            print_to_slack("starting buy and sell")
            print_to_slack("unlocked coins: ", unlock)

            while now < time_checkpoint + datetime.timedelta(hours=1) - datetime.timedelta(seconds=10):
                # current_prices = list(
                #     pyupbit.get_current_price(candidates).values())
                now = datetime.datetime.now()

                for coin_num in range(coin_number):
                    # days = trade_df[:,4][coin_num]
                    # back_length
                    
                    # per_limit = min(3.5,trade_df[:, 2][coin_num]) if (candidates[coin_num] in candidates_high) else trade_df[:, 2][coin_num]
                    # per_limit = min(2.5,trade_df[:, 2][coin_num] - 2.5*already_sell[coin_num])
                    per_limit = min(2.5,trade_df[:, 2][coin_num])

                    # splits = trade_df[:,3][coin_num]
                    ma = trade_df[:, 4][coin_num]
                    k = trade_df[:, 5][coin_num]
                    calculated_target = opening_prices[coin_num] + trade_df[:, 10][coin_num] * k

                    target_price = calculated_target/(1+per_limit/100) if already_sell[coin_num] > 0 else (calculated_target if actual_bid_prices[coin_num] == 0 else actual_bid_prices[coin_num])
                                        
                    
                    target_prices[coin_num] = target_price
                    # print_to_slack("target price for ", candidates[coin_num], "is ", target_price)
                    ma5 = trade_df[:, 11][coin_num]
                    tail_num = trade_df[:, 1][coin_num]
                    # rsi_numpy = trade_df[:, 12][coin_num]

                    last_ticks5 = get_recent_tick5(candidates[coin_num])
                    
                    # print_to_slack(orderbook)
                    ask_price = min([x['HighPrice'] for x in last_ticks5])
                    bid_price = max([x['LowPrice'] for x in last_ticks5])
                    bid_price = bid_price if bid_price <= ask_price else (bid_price+ask_price)/2
                    ask_price = ask_price if bid_price <= ask_price else (bid_price+ask_price)/2
                    
                    
                    # if ask_price > 

                    
                    
                    # print_to_slack("ask_price", ask_price, "    bid is", bid_price)
                    rsi_numpy = rsi_ewm_np(pd.Series(np.append(total_df[candidates[coin_num]].iloc[:,[3]].to_numpy().flatten(), bid_price)),   14)[-1]

                    # sell in the middle of the loop once the percentage reaches per_limit
                    # Later on, change target_price to the actual bid price using uuid
                    # percentage = price/target_price
                    # balance = 0 if not (list(filter(lambda d: d['currency'] in [candidates[coin_num][4:]], balances))) else float(list(
                    #     filter(lambda d: d['currency'] in [candidates[coin_num][4:]], balances))[0]['balance'])
                    
                    # balance = 0

                    if too_low[coin_num] ==1:
                        min10_checkpoint = min10_data[candidates[coin_num]].index[-1]

                        if (now >= min10_checkpoint + datetime.timedelta(minutes=20)) & (now < min10_checkpoint + datetime.timedelta(minutes=30)):
                            print_to_slack("now is ", now, "min10 checkpoint is ", min10_checkpoint, "so updating min10_data")
                            update_min10 = pyupbit.get_ohlcv(candidates[coin_num], interval="minute10", count=2)
                            one_update_min10 = update_min10.iloc[1]

                            if update_min10.index[0] == min10_checkpoint:
                                one_update_min10 = update_min10.iloc[1]
                            elif update_min10.index[0] == min10_checkpoint + datetime.timedelta(minutes=10):
                                one_update_min10 = update_min10.iloc[0]

                            min10_data[candidates[coin_num]] = min10_data[candidates[coin_num]].iloc[1:].append(one_update_min10).drop_duplicates()
                            time.sleep(random.randint(1,10)/30)
                            openings_min10[coin_num] = float(pyupbit.get_ohlcv(candidates[coin_num], interval="minute10", count=1)['open'].values)
                            print_to_slack("openings_min10 for ", candidates[coin_num]," is ", openings_min10[coin_num])

                            min10_checkpoint = min10_data[candidates[coin_num]].index[-1]                           
                            print_to_slack("now is ", now, "updated min10 checkpoint is ", min10_checkpoint, "so updating backtesting")

                            maximums_min10 = find_maximum_index(min10_data[candidates[coin_num]], per=6, days=1, splits=8, back_length=50, high_index=1, splits_min=1)
                            print_to_slack("currently backtesting of min 10 for", candidates[coin_num])
                            optimum_check_min10 = np.vstack(maximums_min10)
                            # there might be different optimums that have maximum index, so just pick the first one
                            optimum_loc_min10 = np.where(
                                optimum_check_min10[:, 9] == np.max(optimum_check_min10[:, 9]))[0][0]
                            optimum_min10 = optimum_check_min10[optimum_loc_min10]
                            too_low_backtests[coin_num] = optimum_min10
                            # last_update_10min[coin_num] = min10_checkpoint.minute
                            print_to_slack("10min backtest results for ", candidates[coin_num], " checkpoint at ",  min10_checkpoint, " is ", too_low_backtests[coin_num])


                        elif now < min10_checkpoint + datetime.timedelta(minutes=20):
                            per_limit_min10 = min(2.5,too_low_backtests[coin_num][2])

                            # splits = trade_df[:,3][coin_num]
                            ma_min10 = too_low_backtests[coin_num][4]
                            k_min10 = too_low_backtests[coin_num][5]
                            calculated_target_min10 = openings_min10[coin_num] + too_low_backtests[coin_num][10] * k

                            target_price_min10 = calculated_target_min10/(1+per_limit_min10/100) if already_sell[coin_num]==1 else (calculated_target_min10 if actual_bid_prices[coin_num] == 0 else actual_bid_prices[coin_num])
                                                
                            
                            target_prices_min10[coin_num] = target_price_min10
                            # print_to_slack("target price for ", candidates[coin_num], "is ", target_price)
                            ma5_min10 = too_low_backtests[coin_num][11]
                            tail_num_min10 =  too_low_backtests[coin_num][1]
                            rsi_numpy_min10 = rsi_ewm_np(pd.Series(np.append(min10_data[candidates[coin_num]].iloc[:,[3]].to_numpy().flatten(), bid_price)),   14)[-1]

                            if bid_price > (1+per_limit_min10/100)*target_price_min10:
                                # balance = upbit.get_balance(candidates[coin_num])
                                if (bought[coin_num] == 1) & (current_balances[coin_num]*actual_bid_prices[coin_num] > 5000):
                                    upbit.sell_market_order( candidates[coin_num], current_balances[coin_num])
                                    print_to_slack("-------Sold coin ", candidates[coin_num], " at", bid_price, " because profit exceeded percentage_min10 cutoff: ", per_limit_min10)
                                    already_sell[coin_num] = 1
                                    bought[coin_num] = 0
                                    current_balances[coin_num] = 0
                                    actual_bid_prices[coin_num] = 0

                            elif bid_price <= (1+per_limit_min10/100)*target_price_min10:
                                if bought[coin_num] == 0:
                                    if ma_min10:
                                        if (ask_price >= target_price_min10) & (openings_min10[coin_num] >= ma5_min10) & (rsi_numpy_min10 >= 50):
                                            if balance_kr > 5000:
                                                upbit.buy_market_order(
                                                    candidates[coin_num], (balance_kr/max(2,coin_number))/(1+fee/100))    #divide the whole seed money

                                                print_to_slack("-------bought coin min10 ", candidates[coin_num], " wanted price at ", ask_price)
                                                bought[coin_num] = 1
                                                already_sell[coin_num] = 0
                                                # current_balances[coin_num] = upbit.get_balance(candidates[coin_num])
                                                # print_to_slack(bought, actual_bid_prices, current_balances)
                                            
                                    elif not(ma_min10):
                                        if (ask_price >= target_price_min10) & (rsi_numpy_min10 >= 50) :
                                            if balance_kr > 5000:
                                                upbit.buy_market_order(
                                                    candidates[coin_num], (balance_kr/max(2,coin_number))/(1+fee/100))    #divide the whole seed money
                                                
                                                print_to_slack("-------bought coin min10 ", candidates[coin_num], " wanted price at ", ask_price)
                                                bought[coin_num] = 1
                                                already_sell[coin_num] = 0
                                                # current_balances[coin_num] = upbit.get_balance(candidates[coin_num])
                                                # print_to_slack(bought, actual_bid_prices, current_balances)

                                
                                elif bought[coin_num]==1:
                                    # if (bid_price <= openings_min10[coin_num]*(2-actual_bid_prices[coin_num]/openings_min10[coin_num])) & (current_balances[coin_num]*actual_bid_prices[coin_num] > 5000):
                                    # if (bid_price <= openings_min10[coin_num]) & (current_balances[coin_num]*actual_bid_prices[coin_num] > 5000):    
                                    if (bid_price <= actual_bid_prices[coin_num]*0.95) & (current_balances[coin_num]*actual_bid_prices[coin_num] > 5000):
                                        upbit.sell_market_order( candidates[coin_num], current_balances[coin_num])
                                        # print_to_slack("-------Sold coin ", candidates[coin_num], " because prices got lower than opening price_min10 * (1-percentage) : ", actual_bid_prices[coin_num]/openings_min10[coin_num] -1)
                                        print_to_slack("-------Sold coin ", candidates[coin_num], " because prices got too lower than target_min10")
                                        # already_sell[coin_num] = 1
                                        # too_low[coin_num] = 1
                                        bought[coin_num] = 0
                                        current_balances[coin_num] = 0
                                        actual_bid_prices[coin_num] = 0 

                                    else:
                                        pass


                    
                    elif too_low[coin_num]==0:

                        if bid_price > (1+per_limit/100)*target_price:
                            # balance = upbit.get_balance(candidates[coin_num])
                            # counting = 1 if already_sell[coin_num] == 0 else already_sell[coin_num]+2
                            # if (bought[coin_num] == 1) & (bid_price >= target_price*((1+per_limit/100)**counting)) & (current_balances[coin_num]*actual_bid_prices[coin_num] > 5000):
                            if (bought[coin_num] == 1) & (current_balances[coin_num]*actual_bid_prices[coin_num] > 5000):
                                upbit.sell_market_order( candidates[coin_num], current_balances[coin_num])
                                already_sell[coin_num] += 1
                                print_to_slack("-------Sold coin ", candidates[coin_num], " at", bid_price, " because profit exceeded percentage cutoff: ", per_limit, "already sell: ", already_sell[coin_num], "target : ", target_price)
                                bought[coin_num] = 0
                                current_balances[coin_num] = 0
                                actual_bid_prices[coin_num] = 0 
                                # print_to_slack(bought, actual_bid_prices, current_balances)
                            
                            # elif (bought[coin_num]==0) & (ask_price > target_price*((1+per_limit/100)**(already_sell[coin_num]+1))):
                            #     upbit.buy_market_order(
                            #     candidates[coin_num], (balance_kr/max(2,coin_number))/(1+fee/100))
                            #     print_to_slack("-------bought coin", candidates[coin_num], " wanted price at ", ask_price, "profit of ", already_sell[coin_num] +1, " times percent")
                            #     bought[coin_num] = 1
                             
                                
                            

                        # if percentage did not exceed the cutoff, sell in at the last 10secs loop
                        # and now we buy if the price matches our condition

                        # islastvolume_increased = (total_df[candidates[coin_num]]['volume'][-1] > total_df[candidates[coin_num]]['volume'][-2])
                        elif bid_price <= (1+per_limit/100)*target_price:

                            if bought[coin_num] == 0:
                                if candidates[coin_num] in candidates_high:
                                    if ma:
                                        if (ask_price >= target_price) & (opening_prices[coin_num] >= ma5) & (rsi_numpy >= 50):
                                            if (balance_kr > 5000) & bool(unlock[coin_num]):
                                                upbit.buy_market_order(
                                                    candidates[coin_num], (balance_kr/max(2,coin_number))/(1+fee/100))    #divide the whole seed money

                                                print_to_slack("-------bought coin", candidates[coin_num], " wanted price at ", ask_price)
                                                bought[coin_num] = 1
                                                already_sell[coin_num] = 0
                                                # current_balances[coin_num] = upbit.get_balance(candidates[coin_num])
                                                # print_to_slack(bought, actual_bid_prices, current_balances)
                                            
                                    elif not(ma):
                                        if (ask_price >= target_price) & (rsi_numpy >= 50) :
                                            if (balance_kr > 5000) & bool(unlock[coin_num]) :
                                                upbit.buy_market_order(
                                                    candidates[coin_num], (balance_kr/max(2,coin_number))/(1+fee/100))    #divide the whole seed money
                                                
                                                print_to_slack("-------bought coin", candidates[coin_num], " wanted price at ", ask_price)
                                                bought[coin_num] = 1
                                                already_sell[coin_num] = 0
                                                # current_balances[coin_num] = upbit.get_balance(candidates[coin_num])
                                                # print_to_slack(bought, actual_bid_prices, current_balances)
                                
                                elif candidates[coin_num] in candidates_low:   ## not using rsi_numpy in condition if coin is in candidates_low
                                    if ma:
                                        if (ask_price >= target_price) & (opening_prices[coin_num] >= ma5) :
                                            if (balance_kr > 5000) & bool(unlock[coin_num]):
                                                upbit.buy_market_order(
                                                    candidates[coin_num], (balance_kr/max(2,coin_number))/(1+fee/100))    #divide the whole seed money
                                                
                                                print_to_slack("-------bought coin", candidates[coin_num], " wanted price at ", ask_price)

                                                bought[coin_num] = 1
                                                already_sell[coin_num] = 0
                                                # current_balances[coin_num] = upbit.get_balance(candidates[coin_num])
                                                # print_to_slack(bought, actual_bid_prices, current_balances)
                                            
                                    elif not(ma):
                                        if (ask_price >= target_price) :
                                            if (balance_kr > 5000) & bool(unlock[coin_num]):
                                                upbit.buy_market_order(
                                                    candidates[coin_num], (balance_kr/max(2,coin_number))/(1+fee/100))    #divide the whole seed money
                                                
                                                # balancebook = upbit.get_balances()
                                                # # if type(balancebook) == list:
                                                # #     pass
                                                # # else:
                                                # #     while type(balancebook) != list:
                                                # #         time.sleep(max(coin_number/13, gap.total_seconds()))
                                                # #         balancebook = pyupbit.get_orderbook(tickers=candidates[coin_num])

                                                # try :
                                                #     actual_bid_prices[coin_num] = list(filter(lambda d: d['currency'] in [candidates[coin_num][4:]], balancebook))[0]['avg_buy_price']

                                                # except :
                                                #     # time.sleep(max(coin_number/10, gap.total_seconds()))
                                                #     balancebook = upbit.get_balances()
                                                #     actual_bid_prices[coin_num] = list(filter(lambda d: d['currency'] in [candidates[coin_num][4:]], balancebook))[0]['avg_buy_price']

                                                # print_to_slack("-------bought coin", candidates[coin_num], " wanted price at ", actual_bid_prices[coin_num])
                                                print_to_slack("-------bought coin", candidates[coin_num], " wanted price at ", ask_price)

                                                bought[coin_num] = 1
                                                already_sell[coin_num] = 0
                                                # current_balances[coin_num] = upbit.get_balance(candidates[coin_num])
                                                # print_to_slack(bought, actual_bid_prices, current_balances)
                                # print_to_slack("end of loop")
                            elif bought[coin_num] == 1: # try to sell if  getting to low, and apply same rule in 10-min scope
                                if already_sell[coin_num] == 0:
                                    # if (bid_price <= opening_prices[coin_num]*(2-actual_bid_prices[coin_num]/opening_prices[coin_num])) & (current_balances[coin_num]*actual_bid_prices[coin_num] > 5000):
                                    if (bid_price <= actual_bid_prices[coin_num]*0.95) & (current_balances[coin_num]*actual_bid_prices[coin_num] > 5000):    
                                        upbit.sell_market_order( candidates[coin_num], current_balances[coin_num])
                                        print_to_slack("-------Sold coin ", candidates[coin_num], " because prices got lower than actual_bid_prices[coin_num]*0.95 : ", actual_bid_prices[coin_num], bid_price)
                                        # already_sell[coin_num] = 1
                                        too_low[coin_num] = 1
                                        bought[coin_num] = 0
                                        current_balances[coin_num] = 0
                                        actual_bid_prices[coin_num] = 0 

                                        print_to_slack("-------getting last 100 data of ", candidates[coin_num], " in a scope of 10 minutes")
                                        min10_data[candidates[coin_num]] = pyupbit.get_ohlcv(candidates[coin_num], interval="minute10", count = 100)[:-1]
                                        if int(min10_data[candidates[coin_num]].index[-1].minute/10) == int(now.minute/10)-1:
                                            pass
                                        elif int(min10_data[candidates[coin_num]].index[-1].minute/10) == int(now.minute/10)-2:
                                            min10_data[candidates[coin_num]] = pyupbit.get_ohlcv(candidates[coin_num], interval="minute10", count = 100)[:-1]
                                        
                                        print_to_slack("last checkpoint of min10 data is ", min10_data[candidates[coin_num]].index[-1])
       
                                        maximums_min10 = find_maximum_index(min10_data[candidates[coin_num]], per=6, days=1, splits=8, back_length=50, high_index=1, splits_min=1)
                                        print_to_slack("first backtesting of min 10 for", candidates[coin_num])
                                        optimum_check_min10 = np.vstack(maximums_min10)
                                        # there might be different optimums that have maximum index, so just pick the first one
                                        optimum_loc_min10 = np.where(
                                            optimum_check_min10[:, 9] == np.max(optimum_check_min10[:, 9]))[0][0]
                                        optimum_min10 = optimum_check_min10[optimum_loc_min10]
                                        too_low_backtests[coin_num] = optimum_min10
                                        # last_update_10min[coin_num] = min10_checkpoint.minute
                                        print_to_slack("10min backtest results for ", candidates[coin_num], " is ", too_low_backtests[coin_num])    
                                    else:
                                        pass

                                elif already_sell[coin_num] > 0:
                                    # bought_price = target_price*((1+per_limit/100)**(already_sell[coin_num]+1))
                                    if current_balances[coin_num]*actual_bid_prices[coin_num] > 5000:                     
                                        upbit.sell_market_order( candidates[coin_num], current_balances[coin_num])
                                        print_to_slack("-------Sold coin ", candidates[coin_num], " because prices got lower than original target, after selling already: ", already_sell[coin_num])
                                        already_sell[coin_num] = 0
                                        too_low[coin_num] = 0
                                        bought[coin_num] = 0
                                        current_balances[coin_num] = 0
                                        actual_bid_prices[coin_num] = 0                                     


                # need this to break from current while loop, go to the selling section and proceed to next hour
                gap = datetime.datetime.now() - now                
                # print_to_slack(gap)

                time.sleep(min(coin_number/13,gap.total_seconds()))
                if(now.minute % 5 == 0) & (now.second < 2):
                    print_to_slack("---time is now: ", now, " & price monitoring ")
                    print_to_slack("-> targets are: ", target_prices)
                    print_to_slack("-> bought: ",bought)
                    print_to_slack("-> actual_bid_price: ",actual_bid_prices)
                    print_to_slack("-> current_balances: ",current_balances)
                    print_to_slack("-> already sell: ",already_sell)
                    if (sum(too_low)>0):
                        print_to_slack("-> 10min targest are ", target_prices_min10)
                    print_to_slack(" ")
                else:
                    if now.second % 1 == 0:
                        balances = upbit.get_balances()
                        # if (len(balances) > 1) & (len(balances) == sum(bought) + 1):
                        if len(balances) > 1:
                            updated = np.where(bought==1)[0]
                            for num in range(updated.size):
                                coin_num = updated[num]
                                current_balances[coin_num] = list(filter(lambda d: d['currency'] in [candidates[coin_num][4:]], balances))[0]['balance'] if not (candidates[coin_num]=="KRW-BTC") else list(filter(lambda d: d['currency'] in [candidates[coin_num][4:]], balances))[0]['balance'] - 0.01059596
                                actual_bid_prices[coin_num] = list(filter(lambda d: d['currency'] in [candidates[coin_num][4:]], balances))[0]['avg_buy_price'] if not (candidates[coin_num]=="KRW-BTC") else ((0.01059596+current_balances[coin_num])*list(filter(lambda d: d['currency'] in [candidates[coin_num][4:]], balances))[0]['avg_buy_price']/current_balances[coin_num]) - 450191
            
        elif coin_number == 0:
            print_to_slack("no candidates this time, current time: ", datetime.datetime.now() ,"// checkpoint is ", time_checkpoint)
            waitingfor = (time_checkpoint + datetime.timedelta(hours=1) - datetime.timedelta(seconds=5) - datetime.datetime.now()).total_seconds()
            print_to_slack("waiting for ", waitingfor)
            time.sleep(waitingfor)
            now = datetime.datetime.now()

    elif now < time_checkpoint + datetime.timedelta(hours=1):
        while now < time_checkpoint + datetime.timedelta(hours=1):
            if sum(bought) >0:
                for coin_num in range(coin_number):
                    if bought[coin_num] == 1 :
                        # if already_sell[coin_num] == 1:
                        #     print_to_slack("already sell & bought is both 1 at the same time, sth might be wrong")
                        # elif already_sell[coin_num] == 0:
                            # balance_coin = upbit.get_balance(candidates[coin_num])
                        if actual_bid_prices[coin_num] * current_balances[coin_num] > 5000:
                            time.sleep(0.75)                                
                            upbit.sell_market_order(candidates[coin_num], current_balances[coin_num])
                            print_to_slack("-------coin ", candidates[coin_num], " is sold!-------")
                            bought[coin_num] = 0

                    elif bought[coin_num] == 0 :
                        if already_sell[coin_num] > 0:
                            print_to_slack("---coin ", candidates[coin_num], " was already sold because the price got greater than percentage cutoff")
                        else: 
                            print_to_slack("---coin", candidates[coin_num], " never bought, so cannot sell ")
                    else:
                        print_to_slack ("something wrong somehow")
                        
            elif sum(bought) ==0:
                wating_final = time_checkpoint + datetime.timedelta(hours=1) - now 
                print_to_slack("---all coins are sold, wating for next loop: ", wating_final)
                time.sleep((wating_final).total_seconds())

            now = datetime.datetime.now()

        time_checkpoint = time_checkpoint + datetime.timedelta(hours=1)

        if time_checkpoint == total_df['KRW-BTC'].index[-1] + datetime.timedelta(hours=1):
            print_to_slack("Time is well coordinated")

        
    else:
        print_to_slack("SOMETHING might be WRONG if this msg repeats!!!")
        time.sleep(1)
        now = datetime.datetime.now()
        time_checkpoint = time_checkpoint + datetime.timedelta(hours=1)

    



    # except Exception as e:
    #     print(e)

    # time.sleep(1)

