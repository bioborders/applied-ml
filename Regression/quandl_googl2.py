import pandas as pandas
import quandl
import math
import config

quandl.ApiConfig.api_key = config.quandle_api_key

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_percent'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['percent_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close','HL_percent','percent_change','Adj. Volume']]

forecast_column = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1*len(df)))

df['label'] = df[forecast_column].shift(-forecast_out)

print(df.head())
