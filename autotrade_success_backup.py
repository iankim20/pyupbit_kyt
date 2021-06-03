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


def rsi(df, cutoff):
    # https://towardsdatascience.com/a-possible-trading-strategy-technical-analysis-with-python-ee1168b5f117
    data = df.copy()

    # rsi_period = cutoff
    # chg = data['close'].diff(1)
    # gain = chg.mask(chg<0,0)
    # data['gain'] = gain
    # loss = chg.mask(chg>0,0)
    # data['loss'] = loss
    # avg_gain = gain.ewm(com = rsi_period - 1, min_periods = rsi_period).mean()
    # avg_loss = loss.ewm(com = rsi_period - 1, min_periods = rsi_period).mean()
    # data['avg_gain'] = avg_gain
    # data['avg_loss'] = avg_loss
    # rs = abs(avg_gain/avg_loss)
    # rsi = 100-(100/(1+rs))

    return pd.DataFrame(np_rsi_plain(data.to_numpy(), cutoff))


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


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def np_draw_frame_ma_per_mean(data, k, per, fee, ma, splits):
    df = data.copy().to_numpy()

    ma5 = np.append(np.zeros(5) + np.nan, moving_average(df[:, 3], 5)[:-1])
    # df['ma5'] = df['close'].rolling(window=5).mean().shift(1)

    # 변동폭 계산
    range_coin = df[:, 1]-df[:, 2]
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
    rsi_numpy = np.append(np.zeros(1) + np.nan, np_rsi_plain(df, 14)[:-1])
    # rsi_numpy = np.append(np.zeros(1) + np.nan, np_rsi_v2(df, 14)[:-1])
    # df['rsi'] = rsi(df,14).shift(1)     #shifted by 1 unit because it is what we refer to when considering at current timepoint

    if ma:
        # df['bull'] = df['open'] > df['ma5']
        # ror(수익률), np.where(조건문, 참일때 값, 거짓일때 값), 사고팔 때 1-fee/100만큼 곱해야 하니까 두번 곱한다.
        ror = np.where((df[:, 1] > target) & (df[:, 1] >= (1+per/100)*target) & (df[:, 0] > ma5) & (rsi_numpy >= 50),
                       (1+per/100)*(1-fee/100)*(1-fee/100),
                       np.where((df[:, 1] > target) & (df[:, 1] < (1+per/100)*target) & (df[:, 0] > ma5) & (rsi_numpy >= 50),
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
        ror = np.where((df[:, 1] > target) & (df[:, 1] >= (1+per/100)*target) & (rsi_numpy >= 50),
                       (1+per/100)*(1-fee/100)*(1-fee/100),
                       np.where((df[:, 1] > target) & (df[:, 1] < (1+per/100)*target) & (rsi_numpy >= 50),
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
    new_rsi = np_rsi_plain(df, 14)[-1]

    return hpr[-1], dd[-1], np.sum(ror < 1), new_range_average, new_ma5, new_rsi


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


def find_maximum_index(data, per, days, splits, back_length):
    df = data.copy()
    # compare_data = pd.DataFrame(columns = [0,1,2,3,4,5,6,7,8])
    compare_data = []

    # per_limit = per+1
    # days_limit = days+1
    # splits_limit = splits+1

    for days in np.arange(1, days+1, 1):
        using_data = pd.DataFrame(df.iloc[::days, :][-back_length:])

        length = len(using_data)
        for per in np.arange(2, per+1, 0.5):
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
                        hpr, dd, num_ror_below1, new_range_average, new_ma5, new_rsi = np_draw_frame_ma_per_mean(
                            using_data, k=k, per=per, fee=0.05, ma=ma, splits=splits)
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
                                0.05*5, (num_ror_below1/length)*dd)**0.9, new_range_average, new_ma5, new_rsi])

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

    # do this job evey 1 hour
    if time_checkpoint <= now < time_checkpoint + datetime.timedelta(hours=1) - datetime.timedelta(seconds=10):

        # update total_df by adding data from the last hour
        print("-------updating data--------------")
        tickers = pyupbit.get_tickers(fiat="KRW")
        start = datetime.datetime.now()
        t = timeit.default_timer()

        print("starting time is ", start)
        for i in range(len(tickers)):
            one_coin = pyupbit.get_ohlcv(
                tickers[i], interval="minute60", count=2).iloc[0]
            total_df[tickers[i]] = total_df[tickers[i]
                                            ].iloc[1:].append(one_coin).drop_duplicates()
            time.sleep(random.randint(1, 10)/17)

        end = datetime.datetime.now().time()
        print("ended at ", end, "difference for getting data is ",
              timeit.default_timer() - t)

        print("-------getting high rsi candidates--------------")

        # get high rsi candidates
        coins_rsi = {}
        coin_number = 0
        tail_minimum = 24       # minimum back testing time frame is 24 hours
        while coin_number < 1:
            for coin in total_df:
                coins_rsi[coin] = float(
                    rsi(total_df[coin], 14).tail(tail_minimum).mean().values)
                print("got rsi for", coin)

            high_rsi = {key: value for (
                key, value) in coins_rsi.items() if coins_rsi[key] >= 50}
            coin_number = len(high_rsi)

            tail_minimum += 1

        candidates = [*high_rsi]

        if coin_number > 6:  # if there are too may candidates, filter them to 6 biggest rsi
            candidates = [a_tuple[0] for a_tuple in sorted(
                high_rsi.items(), key=lambda x: x[1], reverse=True)[:6]]
            # coin_number = 6

        opening_prices = []

        print("-------candidates are", candidates)
        coin_number = len(candidates)

        # get optimal values of days, per, splits, ma, k     and opening prices
        trade_df = np.empty([1, 13])
        for coin_num in range(coin_number):
            coin = candidates[coin_num]
            # optimum_check columns: ['days','back_length', 'per','splits','ma','k','hpr','count_loss','mdd', 'vol_index', new_range_avaerage, new_ma5, new_rsi]
            maximums = find_maximum_index(
                total_df[coin], per=11, days=1, splits=5, back_length=14+tail_minimum)
            if maximums:
                optimum_check = np.vstack(maximums)
                # there might be different optimums that have maximum index, so just pick the first one
                optimum_loc = np.where(
                    optimum_check[:, 9] == np.max(optimum_check[:, 9]))[0][0]
                optimum = optimum_check[optimum_loc]
                trade_df = np.vstack((trade_df, optimum))
            elif not maximums:  # backup just in case when maximums are empty, then increase tail_minimum by 1 to find until maximums are not empty
                new_tail = tail_minimum
                while not maximums:
                    new_tail += 1
                    maximums = find_maximum_index(
                        total_df[coin], per=11, days=1, splits=5, back_length=14+new_tail)
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

        while now < time_checkpoint + datetime.timedelta(hours=1) - datetime.timedelta(seconds=10):
            current_prices = list(
                pyupbit.get_current_price(candidates).values())
            for coin_num in range(coin_number):
                # days = trade_df[:,4][coin_num]
                # back_length
                per_limit = trade_df[:, 2][coin_num]
                # splits = trade_df[:,3][coin_num]
                ma = trade_df[:, 4][coin_num]
                k = trade_df[:, 5][coin_num]
                target_price = opening_prices[coin_num] + trade_df[:, 10][coin_num] * k
                ma5 = trade_df[:, 11][coin_num]
                rsi_numpy = trade_df[:, 12][coin_num]

                price = current_prices[coin_num]

                # sell in the middle of the loop once the percentage reaches per_limit
                # Later on, change target_price to the actual bid price using uuid
                # percentage = price/target_price
                balance = 0 if not (list(filter(lambda d: d['currency'] in [candidates[coin_num][4:]], balances))) else float(list(
                    filter(lambda d: d['currency'] in [candidates[coin_num][4:]], balances))[0]['balance'])
                balance_kr = float(list(filter(lambda d: d['currency'] in ['KRW'], balances))[0]['balance']) - 150000

                if price >= (1+per_limit/100)*target_price:
                    if (bought[coin_num] == 1) & (balance*price > 5000):
                        upbit.sell_market_order(
                            candidates[coin_num], balance)
                        print("-------Sold coin ", candidates[coin_num], " because profit exceeded percentage cutoff: ", per_limit)
                        already_sell[coin_num] = 1

                # if percentage did not exceed the cutoff, sell in at the last 10secs loop
                # and now we buy if the price matches our condition

                if bought[coin_num] == 0:
                    if ma:
                        if (price >= target_price) & (opening_prices[coin_num] > ma5) & (rsi_numpy >= 50):
                            if balance_kr > 5000:
                                upbit.buy_market_order(
                                    candidates[coin_num], (balance_kr/coin_number)/(1+fee/100))    #divide the whole seed money
                                print("-------bought coin", candidates[coin_num])
                                bought[coin_num] = 1
                            
                    elif not(ma):
                        if (price >= target_price) & (rsi_numpy >= 50):
                            if balance_kr > 5000:
                                upbit.buy_market_order(
                                    candidates[coin_num], (balance_kr/coin_number)/(1+fee/100))
                                print("-------bought coin", candidates[coin_num])
                                bought[coin_num] = 1

                elif bought[coin_num] != 0:
                    pass

            # need this to break from current while loop, go to the selling section and proceed to next hour
            now = datetime.datetime.now()
            time.sleep(random.randint(1, 10)/17)
            if (now.minute % 5 == 0) & (now.second < 6):
                print("time is now: ", now, "nothing has happend so far")

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
            
            now = datetime.datetime.now()

        time_checkpoint = time_checkpoint + datetime.timedelta(hours=1)
        
    else:
        print("SOMETHING might be WRONG!!!")

    print("-------now time is ", now, "checkpoint is ", time_checkpoint)

    if time_checkpoint == total_df['KRW-BTC'].index[-1] + datetime.timedelta(hours=1):
        print("Time is well coordinated")

    # except Exception as e:
    #     print(e)

    # time.sleep(1)
