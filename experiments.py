# import pandas
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yfinance as yf

# exchange codes http://finabase.blogspot.com/2014/09/interantional-stock-exchange-codes-for.html

path = './Compustat data4.csv.nosync.csv'
df = pd.read_csv(path, sep=',')

# Filters

print(len(df))
# drop all df rows where exchg is [0, 1, 2, 19]
df = df[(df['exchg'] != 0) & (df['exchg'] != 1) & (df['exchg'] != 2)]

# OTCs - it might be better to keep them, because if we exclude them, we will be selecting them if they were OTC at year_start but are not anymore, since compustat updates the exchg for each year with the current value
                                  # So if we use the old values not listed as OTC only, we'll be selecting only the OTCs with great performance, which would be misleading
df = df[df['exchg'] != 13]
df = df[df['exchg'] != 19]

# canada
df = df[(df['exchg'] != 7) & (df['exchg'] != 8) & (df['exchg'] != 9)]

# drop rows without market value
df = df[df['mkvalt'].notna()]
print(len(df))

tickers_to_exclude = ['NE', 'EVTL', 'VAL', 'CHK', 'GPOR', 'FYBR', 'CBL', 'TUEM', 'DTLA.P']
df = df[~df['tic'].isin(tickers_to_exclude)]
print(len(df))


# Configurations
year_start = 2013
year_end = 2014
min_tickers_to_select_row = 4000
only_tickers_with_fiscal_year_ending_at = 12


def do_years(year_start, year_end):

    # Functions

    path_items_descriptions = './items_descriptions.csv'
    df_items_descriptions = pd.read_csv(path_items_descriptions, sep=',')
    def get_description(item):
        item = item.upper()
        
        vals = df_items_descriptions[
            (df_items_descriptions['ItemName'] == item ) | 
            (df_items_descriptions['ItemHdr'] == item) |
            (df_items_descriptions['XpressfeedMnemonic'] == item) 
        ]['ItemDesc']
        return vals.values[0] if len(vals) > 0 else 'Not found'

    def get_file_tag(item):
        item = item.upper()
        
        vals = df_items_descriptions[
            (df_items_descriptions['FileTag'] == item ) | 
            (df_items_descriptions['ItemHdr'] == item) |
            (df_items_descriptions['XpressfeedMnemonic'] == item) 
        ]['FileTag']
        return vals.values[0] if len(vals) > 0 else 'Not found'

    def val_for_tic_at_year(tic, year, month=None):
        msft = yf.Ticker(tic)
        hist = msft.history(period="max")
        hist.to_csv('hist.csv')
        hist = pd.read_csv('hist.csv')
        dft = df[(df['tic'] == tic) & (df['fyear'] == year)]
        date = dft['datadate'].values[0]
        if month != None : 
            month += 1
            monthN = month if month <= 12 else month - 12
            date = f'{year if month <= 12 else year+1}{monthN if monthN > 9 else f"0{monthN}" }15'

        date = pd.to_datetime(date, format='%Y%m%d')
        close = None
        if not 'Date' in hist: return None
        for i in range(0, 5):
            if not 'Date' in hist: continue
            dateStr = date.strftime('%Y-%m-%d')
            valsDay = hist[hist['Date'] == dateStr]
            if len(valsDay) > 0:
                close = valsDay['Close'].values[0]
                break
            else:
                date = date + pd.DateOffset(days=1)
        return close

    def market_caps_for_year(year):
        dfyear = df[df['fyear'] == year]
        # return df with mkvalt and tic
        dfyear['price'] = dfyear['prcc_f'] / dfyear['ajex']
        dfyear = dfyear[['tic', 'mkvalt', 'price']]
        #drop nans
        dfyear = dfyear.dropna()
        return dfyear

    columns = df.columns.tolist()
    print(columns)

    df.head()


    tickers = df['tic']
    # get set of unique tickers
    tickers = set(tickers)
    print(f'{len(tickers)} tickers')

    path = 'tickers.csv'
    df_tickers = pd.DataFrame(tickers)
    df_tickers.to_csv(path, index=False)



    def most_frequent_columns(df):
        map_count = {}
        for c in columns:
            map_count[c] = df[c].describe()['count']

        # sort by count
        map_count = {k: v for k, v in sorted(map_count.items(), key=lambda item: item[1], reverse=True)}

        columns_data = []
        for k, v in map_count.items():
            columns_data.append([k, v, get_description(k), get_file_tag(k)])

        df_columns = pd.DataFrame(columns_data, columns=['Column', 'Count', 'Description', 'FileTag'])
        return df_columns

    df_columns = most_frequent_columns(df)

    path = './columns.csv'
    df_columns.to_csv(path, index=False)


    rows_start = df[df['fyear'] == year_start]
    rows_start = rows_start.reset_index(drop=True)

    rows_start = rows_start[rows_start['datadate'] <= int(f'{year_start if only_tickers_with_fiscal_year_ending_at <= 12 else year_start+1}{only_tickers_with_fiscal_year_ending_at}31')]

    tickers_start = rows_start['tic']
    tickers_start = set(tickers_start)
    print(f'{len(tickers_start)} tickers in {year_start}')

    df_columns_start = most_frequent_columns(rows_start)
    path = f'./columns_{year_start}.csv'
    df_columns_start.to_csv(path, index=False)
    df_columns_start

    min_tickers_to_select_row = (4500/4740) * len(rows_start['tic'])
    # select columns with count > min_count
    def select_columns(df, min_count, file_tag):
        vals = df[df_columns_start['FileTag'] == file_tag]
        vals = vals[vals['Count'] > min_count]['Column']
        return vals.values.tolist()

    vals_bal_ann_indl = select_columns(df_columns_start, min_tickers_to_select_row, 'BAL_ANN_INDL')
    vals_is_ann_indl = select_columns(df_columns_start, min_tickers_to_select_row, 'IS_ANN_INDL')
    val_cf_ann = select_columns(df_columns_start, min_tickers_to_select_row, 'CF_ANN')

    print(f'{len(vals_bal_ann_indl)} columns in BAL_ANN_INDL')
    print(f'{len(vals_is_ann_indl)} columns in IS_ANN_INDL')
    print(f'{len(val_cf_ann)} columns in CF_ANN')

    # select all rows in rows_start with columns of vals_bal_ann_indl and vals_is_ann_indl different than NaN
    rows_start_bal_and_is = rows_start[['tic', 'fyear'] + vals_bal_ann_indl + vals_is_ann_indl + val_cf_ann + ['mkvalt', 'prcc_f']]
    rows_start_bal_and_is = rows_start_bal_and_is.dropna()
    rows_start_bal_and_is = rows_start_bal_and_is.reset_index(drop=True)
    # print(len(rows_start_bal_and_is))
    print(f'{len(rows_start_bal_and_is)} rows with all columns in {year_start}')

    # create csv file
    path = f'./rows_{year_start}_bal_ann_indl.csv'
    rows_start_bal_and_is.to_csv(path, index=False)

    tickers = rows_start_bal_and_is['tic']
    data = rows_start_bal_and_is


    # Regressor

    #mlp regressor 
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from sklearn.kernel_ridge import KernelRidge



    import numpy as np

    # split in x and y
    x = data.drop(columns=['tic', 'fyear', 'mkvalt', 'prcc_f'])
    y = data['mkvalt']

    # normalize x and y
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    # y = scaler.fit_transform(y.values.reshape(-1, 1))   

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # MLP Regressor
    mlp_regressor = MLPRegressor(
        learning_rate='adaptive',
        max_iter=1000,
        hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100),
        activation='relu',
        solver='adam',
    )

    mlp_regressor.fit(x_train, y_train)
    score = mlp_regressor.score(x_test, y_test)
    print(f'MLP Regressor Test Score: {score}')

    # mlp_regressor.fit(x, y)
    # score = mlp_regressor.score(x, y)
    # print(f'MLP Regressor Training Score: {score}')


    # Simulation


    #TODO show correct number of tickers
    print(f'Analysing {len(data)} tickers')
    mapT = {}

    market_caps_2021 = market_caps_for_year(year_end)
    market_caps_2020 = market_caps_for_year(year_start)

    # loop through data
    for i in range(len(data)):
        row = data.iloc[i]
        ticker = row['tic']
        x_i = x[i]
        # print(x_i)
        y_i = row['mkvalt']
        # mlp_regressor.predict([x_i])
        p = mlp_regressor.predict([x_i])
        diff = p[0]/y_i - 1
        # print(f'{ticker}: {p} / ${y_i} ({diff})\n')
        if diff > 0:
            mapT[ticker] = diff

    # sort mapT by biggest diff
    mapT = {k: v for k, v in sorted(mapT.items(), key=lambda item: item[1], reverse=True)}

    #get top 30
    top30 = list(mapT.keys())[:30]

    count = 0
    sumAll = 0
    sumAllYf = 0
    sumAllYfJune = 0
    sumAllYfMPlus3 = 0
    sumAllYfMPlus3CompletedByMkvalt = 0 # same as sumAllYfMPlus3, but uses compustat data to fill the values when they're not available
    noyf = []

    for tic in top30:
        vals_2021 = market_caps_2021[market_caps_2021['tic'] == tic]['price'].values
        # we also exclude some tickers here that are presenting a huge change in market cap - they don't have historical prices for 2020 for some reason, doesn't seem like a good idea to use them
        if len(vals_2021) == 0:
            continue
        m2021 = vals_2021[0] 
        m2020 = market_caps_2020[market_caps_2020['tic'] == tic]['price'].values[0] 
        change = m2021/m2020-1
        if np.isnan(change):
            change = (market_caps_2021[market_caps_2021['tic'] == tic]['mkvalt'].values[0] / market_caps_2020[market_caps_2020['tic'] == tic]['mkvalt'].values[0]) - 1
        exchg = df[df['tic'] == tic]['exchg'].values[0]
        print(f'{tic}: {mapT[tic]} => {change} ({exchg})')
        yf2021 =  val_for_tic_at_year(tic, year_end)
        yf2020 = val_for_tic_at_year(tic, year_start)
        yf2021June = val_for_tic_at_year(tic, year_end, month=18)
        yf2020June = val_for_tic_at_year(tic, year_start, month=18)
        yf2021MPlus3 = val_for_tic_at_year(tic, year_end, month=only_tickers_with_fiscal_year_ending_at+3)
        yf2020MPlus3 = val_for_tic_at_year(tic, year_start, month=only_tickers_with_fiscal_year_ending_at+3)
        if not (yf2021 == None or yf2020 == None):
            changeYf = yf2021/yf2020-1
            sumAllYf += changeYf
            print(f'Yahoo Finance: {changeYf}')
        if not (yf2021June == None or yf2020June == None):
            sumAllYfJune += yf2021June/yf2020June-1
        if not (yf2021MPlus3 == None or yf2020MPlus3 == None):
            sumAllYfMPlus3 += yf2021MPlus3/yf2020MPlus3-1
            sumAllYfMPlus3CompletedByMkvalt += yf2021MPlus3/yf2020MPlus3-1
            print(f'Yahoo Finance M+3: {yf2021MPlus3/yf2020MPlus3-1}')
        else:
            sumAllYfMPlus3CompletedByMkvalt += change
            noyf.append(tic)

        sumAll += change
        count += 1
        print()

    change = (sumAll*100)/count
    changeYf = (sumAllYf*100)/(count - len(noyf))
    changeYfJune = (sumAllYfJune)*100/(count - len(noyf))
    chnageYfM3 = (sumAllYfMPlus3*100)/(count - len(noyf))
    changeYfM3C = (sumAllYfMPlus3CompletedByMkvalt*100)/(count)
    changeYfM3Br = ((sumAllYfMPlus3 -1*(len(noyf) + (30-count)))*100)/(count)

    print(f'Average change: {change:.2f}%')
    print(f'Average change yf: {changeYf:.2f}%')
    print(f'Average change yf (Investing only in june after all the fillings have been made public): {changeYfJune:.2f}%')
    print(f'Average change yf M+3: {chnageYfM3:.2f}%')
    print(f'Average change yf M+3 (Completed by Compustat data): {changeYfM3C:.2f}%')
    print(f'Average change yf M+3 (If all delisted went bankrupt): {changeYfM3Br:.2f}%')
    print(f'No yf: {noyf}')
    print(f'Count: {count}')
    print(f'Coverage: {(count)*100/30:.2f}%')

    return [change, changeYf, changeYfJune, chnageYfM3, changeYfM3C, changeYfM3Br, noyf]

sumAll = 0
sumAllBr = 0
count_years = 0

map_year_change = {}

for year_start in range(2000, 2022):
    try:
        print()
        print(f'Analysing {year_start}')
        year_end = year_start + 1
        res = do_years(year_start, year_end)
        sumAll += res[3]
        sumAllBr += res[5]
        count_years += 1
        map_year_change[year_start] = res[3]
    except:
        print(f'Error analysing {year_start}')

print(f'Average change yf M+3: {sumAll/count_years:.2f}%')
print(f'Average change yf M+3 (if all delisted went bankrupt): {sumAllBr/count_years:.2f}%')

print(map_year_change)