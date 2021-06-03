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

access = "dihtIGpo6j9PkhrsxRGRWPTRKrkw13w8fxVze15C"          # 본인 값으로 변경
secret = "a2YBnwtTvJ6UkmyDgUCnJxA9rWqqtDUQcyuluBx1"          # 본인 값으로 변경
upbit = pyupbit.Upbit(access, secret)

print(upbit.get_balance("KRW-BTC"))     # KRW-XRP 조회
print(upbit.get_balance("KRW"))         # 보유 현금 조회
pd.date_range
print(datetime.datetime.now())

balances = upbit.get_balances()
print("-----", balances)

str1 = 'KRW-SC'


print(list(pyupbit.get_current_price(['KRW-SC', "KRW-BTC"]).values()))


balances = upbit.get_balances()
candidates = ['KRW-DOGE', 'KRW-ETC',
              'KRW-ETH', 'KRW-SBD', 'KRW-GAS', 'KRW-DOT']

print(list(pyupbit.get_current_price(candidates).values()))

# for coin_num in range(len(candidates)):
#     balance = 0 if not (list(filter(lambda d: d['currency'] in [candidates[coin_num][4:]], balances))) else float(list(
#         filter(lambda d: d['currency'] in [candidates[coin_num][4:]], balances))[0]['balance'])
    
#     balance_kr = list(filter(lambda d: d['currency'] in ['KRW'], balances))[0]['balance']

#     print(type(balance), type(balance_kr))

current_prices = list(pyupbit.get_current_price(candidates).values())

print(type(current_prices[0]))

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

# total_df = {}
# tickers = pyupbit.get_tickers(fiat="KRW")
# max_number = 4

# for i in range(len(tickers)):
#     one_coin = get_data(tickers[i], interval="minute60", count=200)[:-2]
#     total_df[tickers[i]] = one_coin

# print(candidates)
# print()

# filtered_by_key = {key: total_df[key]['volume'] for key in candidates}
# candidates_high = [a_tuple[0] for a_tuple in sorted(filtered_by_key.items(), key=lambda x: x[1].tail(36).mean(), reverse=True)[:max_number]] 
# volumes = [a_tuple[1].tail(36).mean() for a_tuple in sorted(filtered_by_key.items(), key=lambda x: x[1].tail(36).mean(), reverse=True)[:max_number]] 


# print(candidates_high, volumes)



def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def np_rsi_plain_1D(array,cutoff):
    rsi_period = cutoff
    chg = np.diff(array)
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

def rsi_ewm(data, period):         #input is daaframe, output is numpy
    delta = data.diff()

    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    _gain = up.ewm(com=(period - 1), min_periods=period).mean()
    _loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()

    RS = _gain / _loss
    return (100 - (100 / (1 + RS))).to_numpy().flatten()

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
    

# total_df = {}
# tickers = pyupbit.get_tickers(fiat="KRW")

# start = datetime.datetime.now()
# t = timeit.default_timer()

# print("starting time is ", start)
# for i in range(10):
#     one_coin = get_data(tickers[i], interval="minute60", count=200)[:-2]
#     total_df[tickers[i]] = one_coin
#     time.sleep(random.randint(1, 10)/17)

# end = datetime.datetime.now().time()
# t2 = timeit.default_timer() - t
# print("ended at ", end, "difference for getting data is ", t2)


# time_checkpoint = total_df['KRW-BTC'].index[-1] + datetime.timedelta(hours=2)
# trade_df = np.empty([1, ??? ])

# run code in an hour unit
# while True:
    # try:
now = datetime.datetime.now()

# do this job evey 1 hour
# if time_checkpoint <= now < time_checkpoint + datetime.timedelta(hours=1) - datetime.timedelta(seconds=10):

#     # update total_df by adding data from the last hour
# print("-------updating data--------------")
# tickers = pyupbit.get_tickers(fiat="KRW")
# filtered_by_key = {key: total_df[key]['volume'][-1]*total_df[key]['close'][-1] for key in tickers}
# tickers = [a_tuple[0] for a_tuple in sorted(filtered_by_key.items(), key=lambda x: x[1])] 

# print(type(tickers))

opening_prices = []
opening_prices.append(float(pyupbit.get_ohlcv("KRW-SBD", interval="minute60", count=1)['open'].values))
print(opening_prices)


target_prices = np.zeros(5)
target_prices[3] = 3
print(datetime.timedelta(milliseconds=500))


hey = pyupbit.get_orderbook(tickers=["KRW-BTC","KRW-XLM"])[0]["orderbook_units"][0]["ask_price"]

candidates = ["KRW-BTC","KRW-XRP", "KRW-EDR"]

for coin_num in range(len(candidates)):
    now = datetime.datetime.now()
    orderbook = 0
    while True:
        try: 
            orderbook = pyupbit.get_orderbook(tickers=candidates[coin_num])
            break
        # print(orderbook)
        except (requests.exceptions.ConnectionError, json.decoder.JSONDecodeError):
            # e = sys.exc_info()[0]
            # orderbook =0
            time.sleep(max(2/15, gap.total_seconds()))
            orderbook = pyupbit.get_orderbook(tickers=candidates[coin_num])
        else:
            pass
    
    gap = datetime.datetime.now()

balances = upbit.get_balances()
# print(filter(lambda d: d['currency'] in ["KRW-SBD"[4:]], upbit.get_balances()))
bought = np.zeros(6)

bought[2] = 1
bought[1] = 0
bought[3] =1
updated = np.where(bought==1)[0]
print(updated.size)


candidates = ['KRW-d',"KRW-b","KRW-s","KRW-dsd"]

# print(rsi_ewm_np(total_df['KRW-BTC'].iloc[:,[3]], 14))

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
        for per in np.arange(0.9, per+1, 0.6):
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

                    for k in np.arange(0.3, 2.0, 0.1):
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


check = ["a","b"]
check.append("c")
print(int((now.minute+7)/10))

too_low_backtests = np.zeros(shape=(3, 12))

data = pyupbit.get_ohlcv("KRW-PLA", interval="minute60", count=100)[:-2]

maximums = find_maximum_index(data, per=10, days=1, splits=9, back_length=16, high_index=1, splits_min=1) ##### 100 or 14+tail_minimum?

optimum_check = np.vstack(maximums)
# there might be different optimums that have maximum index, so just pick the first one
optimum_loc = np.where(optimum_check[:, 9] == np.max(optimum_check[:, 9]))[0][0]
optimum = optimum_check[optimum_loc]
too_low_backtests[2] = optimum
print(too_low_backtests[2][4])
print(too_low_backtests)

def post_message(*values):
    """슬랙 메시지 전송"""
    myToken = "xoxb-2120785924737-2120807096337-Omv2JW7ryhY6QBAuR0zJszGZ"

    for a in values:
        response = requests.post("https://slack.com/api/chat.postMessage",
            headers={"Authorization": "Bearer "+myToken},
            data={"channel": '#online_kyt',"text": values}
        )
        print(a)

number =2
post_message("hey", number)