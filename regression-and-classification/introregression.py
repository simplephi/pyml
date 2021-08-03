import pandas as pd
import quandl
import math
import numpy as np
import pandas_datareader.data as web
import datetime as dt

# Pake matplotlib 3.2 untuk pake pyplot
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pickle

df = quandl.get('WIKI/GOOGL')
# df = pd.read_csv('tsla.csv')

# df = df[['High', 'Low', 'Open', 'Volume', 'Adj Close']]
# df['HL_PCT'] = (df['High'] - df['Adj Close']) / df['Adj Close'] * 100.0
# df['PCT_change'] = (df['Adj Close'] - df['Open']) / df['Open'] * 100.0
# df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]
# forecast_col = 'Adj Close'

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

#           price         x            x            x
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

# 0.01 untuk 1% (ini bisa diganti) dan dibulatkan dari panjang dataframe. Kita pake 0.01 % berarti kita amnil 35 data untuk kedepan nya kita mau prediksi
forecast_out = int(math.ceil(0.1*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

# Preprosesing
X = np.array(df.drop(['label'], 1))

# Normalisasi agar tidak terlalu besar
X = preprocessing.scale(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]


df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Bisa ganti SVM dll, dan bisa juga tambah (n_jobs=-1) untuk sesuaikan dengan memory nya kita
clf = LinearRegression()
clf.fit(X_train, y_train)

# Build Model nya
# with open('linearregression.pickle', 'wb') as f:
#     pickle.dump(clf, f)

# pickle_in = open('linearregression.pickle', 'rb')
# clf = pickle.load(pickle_in)


accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

# Visualisasi

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = dt.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

# list values di np.nan dari semua label dan +[i] itu forecast nya
# df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

# untuk lihat data prediksi nya dibawah 
print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


