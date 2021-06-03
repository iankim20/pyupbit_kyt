import pyupbit
import pandas as pd
import numpy as np
import os
import datetime
import time
import random
import timeit
# import logging 

import sys
# old_stdout = sys.stdout

# log_file = open("output.log","w")
# sys.stdout = log_file

# sys.stdout = old_stdout
# log_file.close()



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


def find_maximum_index(data, per, days, splits, back_length, high_index):
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
            for splits in np.arange(1, splits+1, 1):
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

print("starting time is ", start)
for i in range(len(tickers)):
    one_coin = get_data(tickers[i], interval="minute60", count=200)[:-2]
    total_df[tickers[i]] = one_coin
    time.sleep(random.randint(1, 10)/17)

end = datetime.datetime.now().time()
t2 = timeit.default_timer() - t
print("ended at ", end, "difference for getting data is ", t2)


time_checkpoint = total_df['KRW-BTC'].index[-1] + datetime.timedelta(hours=2)
# trade_df = np.empty([1, ??? ])

# run code in an hour unit
while True:
    # try:
    now = datetime.datetime.now()
    print("-------now time is ", now, "checkpoint is ", time_checkpoint)
    print(" ")
    print(" ")

    # do this job evey 1 hour
    if time_checkpoint <= now < time_checkpoint + datetime.timedelta(hours=1) - datetime.timedelta(seconds=10):

        # update total_df by adding data from the last hour
        print("-------updating data--------------")
        tickers = pyupbit.get_tickers(fiat="KRW")
        filtered_by_key = {key: total_df[key]['volume'][-1]*total_df[key]['close'][-1] for key in tickers}
        tickers = [a_tuple[0] for a_tuple in sorted(filtered_by_key.items(), key=lambda x: x[1])]  #filter from lower volume to high, in order to prevent update loss (although we check timecheckpoint)

        print(tickers)
        start = datetime.datetime.now()
        t = timeit.default_timer()

        print("starting time is ", start)
        for i in range(len(tickers)):
            # isit_uptodate = time_checkpoint
            # while isit_uptodate != (time_checkpoint - datetime.timedelta(hours=1)):            #initial value would be 1 hour ago beacuse when it was called last was before loop
            #     one_coin_data = pd.DataFrame(pyupbit.get_ohlcv(tickers[i], interval="minute60", count=2))
            #     print(one_coin_data)
            #     one_coin = one_coin_data.iloc[0]
            #     isit_uptodate = one_coin.name
            one_coin_data = pyupbit.get_ohlcv(tickers[i], interval="minute60", count=2)   #due to delayed update, this might be either [now-2,now-1] or [now-1, now] so we need to filter

            if one_coin_data.index[-1] == (time_checkpoint - datetime.timedelta(hours=1)):   #if [now-2, now-1]
                one_coin = one_coin_data.iloc[1]
                total_df[tickers[i]] = total_df[tickers[i]].iloc[1:].append(one_coin).drop_duplicates()
                if tickers[i] in ["KRW-BTC","KRW-MTL","KRW-NEO","KRW-IOST","KRW-EFL", "KRW-ETH", "KRW-ARK"]:
                    print(total_df[tickers[i]].iloc[[0,1,-2,-1]])
                    print(" ")
                    print(pd.DataFrame(one_coin).T)                

            elif one_coin_data.index[-1] == time_checkpoint:   #if [now-1, now]
                one_coin = one_coin_data.iloc[0]
                total_df[tickers[i]] = total_df[tickers[i]].iloc[1:].append(one_coin).drop_duplicates()
                if tickers[i] in ["KRW-BTC","KRW-MTL","KRW-NEO","KRW-IOST","KRW-EFL", "KRW-ETH", "KRW-ARK"]:
                    print(total_df[tickers[i]].iloc[[0,1,-2,-1]])
                    print(" ")
                    print(pd.DataFrame(one_coin).T)

            else:
                print("something's wrong with updating")                  

            time.sleep(random.randint(1, 10)/30)


        end = datetime.datetime.now().time()
        print({key: len(total_df[key]['volume']) for key in tickers})
        print("ended at ", end, "difference for getting data is ",
              timeit.default_timer() - t)

        #-----------------------------------------------------------------------------------
        print("-------getting 4 high rsi candidates (tail minimum1 is 2)--------------")

        # get high rsi candidates
        coins_rsi = {}
        coin_number = 0
        high_rsi_cutoff = 60
        tail_minimum1 = 2       # minimum back testing time frame is 24 hours
        # total_backup = total_df.copy()

        while high_rsi_cutoff >= 55:
            while tail_minimum1 <=5:
                    
                for coin in total_df:
                    coins_rsi[coin] = rsi_ewm_np(total_df[coin].iloc[:,[3]], 14)[-tail_minimum1:].mean()
                    # print(tail_minimum1, "got high rsi for", coin)

                high_rsi = {key: value for (
                    key, value) in coins_rsi.items() if (coins_rsi[key] >= high_rsi_cutoff) & (total_df[key]['volume'][-1] >= total_df[key]['volume'][-2])}
                coin_number = len(high_rsi)
                if coin_number > 0:
                    break
                else:
                    print(tail_minimum1, "got high rsi for all coins with high rsi cutoff", high_rsi_cutoff)
                    tail_minimum1 += 1  

            high_rsi_cutoff -= 0.5
            tail_minimum1 =2 
            if coin_number > 0:
                break
            

        candidates_high = [*high_rsi]
        print("before filtering by capacity, candidates_high are ", candidates_high)
        print(coins_rsi)

        max_number = 4 ######## maximum number of coins to bid, sort by rsi index 

        if coin_number > max_number:  # if there are too may candidates, filter them to 6 biggest rsi
            # candidates_high = [a_tuple[0] for a_tuple in sorted(
            #     high_rsi.items(), key=lambda x: x[1], reverse=True)[:max_number]]
            filtered_by_key = {key: total_df[key]['volume']*total_df[key]['close'] for key in candidates_high}
            candidates_high = [a_tuple[0] for a_tuple in sorted(filtered_by_key.items(), key=lambda x: x[1].tail(14+tail_minimum1).mean(), reverse=True)[:max_number]]  #filter high rsi candidates by volume size

            # coin_number = 6
        
        print("high rsi candidates are ", candidates_high)
        print({key:total_df[key].iloc[[0,1,2,-3,-2,-1]] for key in candidates_high})
        #------------total
        #-----------------------------------------------------------------------------------
        print("-------getting 4 low rsi(with high volume, tail_minimum2=30) candidates--------------")

        # get high rsi candidates
        coins_rsi = {}
        coin_number = 0
        low_rsi_cutoff = 45
        tail_minimum2 = 30
        # total_backup = total_df.copy()

        while low_rsi_cutoff <= 50:
            while tail_minimum2 <=140:
                    
                for coin in total_df:
                    coins_rsi[coin] = rsi_ewm_np(total_df[coin].iloc[:,[3]], 14)[-tail_minimum1:].mean()
                    # print(tail_minimum1, "got high rsi for", coin)

                low_rsi = {key: value for (
                    key, value) in coins_rsi.items() if (coins_rsi[key] < low_rsi_cutoff)  & (total_df[key]['volume'][-1] >= total_df[key]['volume'][-2])}
                coin_number = len(low_rsi)

                if coin_number > 0:
                    break
                else:
                    if tail_minimum2 % 20 == 0:
                        print(tail_minimum2, "got low rsi for all coins with low rsi cutoff ", low_rsi_cutoff)
                    tail_minimum2 += 1  

            low_rsi_cutoff += 1
            tail_minimum2 = 30
            if coin_number > 0:
                break



        candidates_low = [x for x in [*low_rsi] if not x in candidates_high]
        print("before filtering by capacity, candidates_low are ", candidates_low)
        print(coins_rsi)

        coin_number = len(candidates_low)

        max_number = 8 - len(candidates_high) ######## maximum number of coins to bid, sort by rsi index 

        if coin_number > max_number:  # if there are too may candidates, filter them to 6 biggest rsi
            filtered_by_key = {key: total_df[key]['volume']*total_df[key]['close'] for key in candidates_low}
            candidates_low = [a_tuple[0] for a_tuple in sorted(filtered_by_key.items(), key=lambda x: x[1].tail(14+tail_minimum2).median(), reverse=True)[:max_number]] 


            # coin_number = 6
        print("low rsi candidates are ", candidates_low, "cutoff is ", low_rsi_cutoff)
        print({key:total_df[key].iloc[[0,1,2,-3,-2,-1]] for key in candidates_low})

        #------------------------------------------------------------------------------------



        opening_prices = []
        candidates = candidates_high + candidates_low

        print("-------candidates are", candidates)
        coin_number = len(candidates)
        # total_backup = total_df.copy()

        # get optimal values of days, per, splits, ma, k     and opening prices
        trade_df = np.empty([1, 12])
        for coin_num in range(coin_number):
            coin = candidates[coin_num]
            # optimum_check columns: ['days','back_length', 'per','splits','ma','k','hpr','count_loss','mdd', 'vol_index', new_range_avaerage, new_ma5]
            maximums = find_maximum_index(
                total_df[coin], per=10, days=1, splits=9, back_length=14+tail_minimum1 if (coin in candidates_high) else 14+tail_minimum2, high_index=1 if (coin in candidates_high) else 0) ##### 100 or 14+tail_minimum?
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
                        total_df[coin], per=11, days=1, splits=9, back_length=new_tail, high_index=1 if (coin in candidates_high) else 0)
                optimum_check = np.vstack(maximums)
                # there might be different optimums that have maximum index, so just pick the first one
                optimum_loc = np.where(
                    optimum_check[:, 9] == np.max(optimum_check[:, 9]))[0][0]
                optimum = optimum_check[optimum_loc]
                trade_df = np.vstack((trade_df, optimum))

            print("backtesting ended for", coin)
            opening_prices.append(float(pyupbit.get_ohlcv(
                coin, interval="minute60", count=1)['open'].values))

        # take away the first array that was only there for initialization
        trade_df = trade_df[1:]
        print("-------trade df is ",trade_df)

        ###### optimizaion (back testing) is all prepared for seeking which ones to buy in this unit of 1 hour #####
        bought = np.zeros(coin_number)
        already_sell = np.zeros(coin_number)
        balances = upbit.get_balances()
        current_prices = []

        print("starting buy and sell")
        
        if coin_number >0 :
            while now < time_checkpoint + datetime.timedelta(hours=1) - datetime.timedelta(seconds=10):
                current_prices = list(
                    pyupbit.get_current_price(candidates).values())
                for coin_num in range(coin_number):
                    # days = trade_df[:,4][coin_num]
                    # back_length
                    
                    per_limit = min(3.5,trade_df[:, 2][coin_num]) if (candidates[coin_num] in candidates_high) else trade_df[:, 2][coin_num]
                    # per_limit = trade_df[:, 2][coin_num]


                    # splits = trade_df[:,3][coin_num]
                    ma = trade_df[:, 4][coin_num]
                    k = trade_df[:, 5][coin_num]
                    target_price = opening_prices[coin_num] + trade_df[:, 10][coin_num] * k
                    ma5 = trade_df[:, 11][coin_num]
                    tail_num = trade_df[:, 1][coin_num]
                    # rsi_numpy = trade_df[:, 12][coin_num]

                    price = current_prices[coin_num]
                    rsi_numpy = rsi_ewm_np(pd.Series(np.append(total_df[candidates[coin_num]].iloc[:,[3]].to_numpy().flatten(), price)),   14)[-1]


                    # sell in the middle of the loop once the percentage reaches per_limit
                    # Later on, change target_price to the actual bid price using uuid
                    # percentage = price/target_price
                    balance = 0 if not (list(filter(lambda d: d['currency'] in [candidates[coin_num][4:]], balances))) else float(list(
                        filter(lambda d: d['currency'] in [candidates[coin_num][4:]], balances))[0]['balance'])
                    balance_kr = float(list(filter(lambda d: d['currency'] in ['KRW'], balances))[0]['balance']) - 100000

                    if price >= (1+per_limit/100)*target_price:
                        if (bought[coin_num] == 1) & (balance*price > 5000):
                            upbit.sell_market_order(
                                candidates[coin_num], balance)
                            print("-------Sold coin ", candidates[coin_num], " because profit exceeded percentage cutoff: ", per_limit)
                            already_sell[coin_num] = 1
                            bought[coin_num] == 0
                            print(bought)

                    # if percentage did not exceed the cutoff, sell in at the last 10secs loop
                    # and now we buy if the price matches our condition

                    # islastvolume_increased = (total_df[candidates[coin_num]]['volume'][-1] > total_df[candidates[coin_num]]['volume'][-2])
                    elif price < (1+per_limit/100)*target_price:
                        if bought[coin_num] == 0:
                            if candidates[coin_num] in candidates_high:
                                if ma:
                                    if (price >= target_price) & (opening_prices[coin_num] >= ma5) & (rsi_numpy >= 50):
                                        if balance_kr > 5000:
                                            upbit.buy_market_order(
                                                candidates[coin_num], (balance_kr/coin_number)/(1+fee/100))    #divide the whole seed money
                                            print("-------bought coin", candidates[coin_num], " wanted price at ", price)
                                            bought[coin_num] = 1
                                            print(bought)
                                        
                                elif not(ma):
                                    if (price >= target_price) & (rsi_numpy >= 50) :
                                        if balance_kr > 5000:
                                            upbit.buy_market_order(
                                                candidates[coin_num], (balance_kr/coin_number)/(1+fee/100))
                                            print("-------bought coin", candidates[coin_num], " wanted price at ", price)
                                            bought[coin_num] = 1
                                            print(bought)
                            
                            elif candidates[coin_num] in candidates_low:   ## not using rsi_numpy in condition if coin is in candidates_low
                                if ma:
                                    if (price >= target_price) & (opening_prices[coin_num] >= ma5) :
                                        if balance_kr > 5000:
                                            upbit.buy_market_order(
                                                candidates[coin_num], (balance_kr/coin_number)/(1+fee/100))    #divide the whole seed money
                                            print("-------bought coin", candidates[coin_num], " wanted price at ", price)
                                            bought[coin_num] = 1
                                            print(bought)
                                        
                                elif not(ma):
                                    if (price >= target_price) :
                                        if balance_kr > 5000:
                                            upbit.buy_market_order(
                                                candidates[coin_num], (balance_kr/coin_number)/(1+fee/100))
                                            print("-------bought coin", candidates[coin_num], " wanted price at ", price)
                                            bought[coin_num] = 1
                                            print(bought)

                        elif bought[coin_num] != 0:
                            pass

                # need this to break from current while loop, go to the selling section and proceed to next hour
                now = datetime.datetime.now()
                time.sleep(random.randint(1, 10)/17)
                if (now.minute % 5 == 0) & (now.second < 1):
                    print("time is now: ", now, "nothing has happend so far")
            
        elif coin_number == 0:
            print("no candidates this time, current time: ", datetime.datetime.now() ,"// checkpoint is ", time_checkpoint)
            waitingfor = (time_checkpoint + datetime.timedelta(hours=1) - datetime.timedelta(seconds=5) - datetime.datetime.now()).total_seconds()
            print("waiting for ", waitingfor)
            time.sleep(waitingfor)
            now = datetime.datetime.now()

    elif now < time_checkpoint + datetime.timedelta(hours=1):
        while now < time_checkpoint + datetime.timedelta(hours=1):
            for coin_num in range(coin_number):
                if (already_sell[coin_num]) == 0 & (bought[coin_num] == 1):
                    balance_coin = upbit.get_balance(candidates[coin_num])
                    if current_prices[coin_num] * balance_coin > 5000:
                        upbit.sell_market_order(candidates[coin_num], balance_coin)
                        print("-------coin ", candidates[coin_num], " is sold!-------")
                elif already_sell[coin_num] == 1:
                    print("---coin ", candidates[coin_num], " was already sold because the price was greater than percentage cutoff")
                elif bought[coin_num] == 0:
                    print("---coin ", candidates[coin_num], " was never bought, so cannot sell")
                else:
                    print("---coin", candidates[coin_num], " pass")
            
            now = datetime.datetime.now()

        time_checkpoint = time_checkpoint + datetime.timedelta(hours=1)

        if time_checkpoint == total_df['KRW-BTC'].index[-1] + datetime.timedelta(hours=1):
            print("Time is well coordinated")

        
    else:
        print("SOMETHING might be WRONG!!!")
        time.sleep(1)
        now = datetime.datetime.now()

    



    # except Exception as e:
    #     print(e)

    # time.sleep(1)

