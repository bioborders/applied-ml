import pandas as pandas
import quandl
import math
import datetime
import config
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

quandl.ApiConfig.api_key = config.quandl_api_key

Stock = 'WIKI/GOOGL'
df = quandl.get(Stock) #choose stock
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_percent'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['percent_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close','HL_percent','percent_change','Adj. Volume']]

forecast_column = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)

df['label'] = df[forecast_column].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression() #01.02.2021: output = 35 days forecast_out; accuracy = 0.9771849294187718
                         #note: LinearRegression(n_jobs=-1 denotes max. threading)
#clf = svm.SVR() #01.02.2021: output = 35 days forecast_out; accuracy = 0.7985706014191183
#clf = svm.SVR(kernel='poly') #01.02.2021: output = 35d forecast_out; accuracy = 0.35345142164613286
#fit = train; score = test

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(accuracy)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

#iterate through forecast set, taking each forecast and then setting those as the values in the dataframe (df)
# .. making those into the 'future' features.

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    # df.loc looks for record in dataframe (i.e. date) - if none exists, it will create one_day
    # np.nan (not a number) for Adj Close, HL_percent etc.
    # +[i] = forecast

#print(df.tail)

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.title(Stock)
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show()

# this clf (classifier) is to be pickled
