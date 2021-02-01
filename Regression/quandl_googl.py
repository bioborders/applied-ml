import pandas as pandas
import quandl

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_percent'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['percent_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close','HL_percent','percent_change','Adj. Volume']]

print(df.head())
