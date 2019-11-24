import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl   
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import product
from datetime import datetime
from datetime import date
warnings.filterwarnings('ignore')
plt.style.use('seaborn-poster')
import plotly.offline as py
import plotly.graph_objs as go
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.callbacks import EarlyStopping
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from random import randint
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTMCell, LSTM
from keras.layers import GRU
from keras.layers import Dense, Activation
from keras.layers import Dropout, Dense 

#read csv data file
df = pd.read_csv('C:\\Users\\Probook6570b\\Documents\\School\\ml\\bitcoin-historical-data\\bitstampUSD_1-min_data_2012-01-01_to_2018-03-27.csv')

#check for non values. false means null values exist
df.isnull().values.any()

#add column Date:  unix timestamp to readable date format:
df['Date'] = pd.to_datetime(df['Timestamp'],unit='s')

# Cleaning our data 
# when Volume_(BTC) is null(None) chamge to 0
df.loc[df['Volume_(BTC)']==None,'Volume_(BTC)']=0.0
# convert Volume_BTC to int
df['Volume_(BTC)'] = df['Volume_(BTC)'].astype('int64')
    
#print(df.head(100))
#print() 
#print() 
#print(df.tail(50))

print("printing col info")
print(df.info())

#get mean weighted price by year
group = df.groupby(df.Date.dt.year)
Yearly_Price = group['Weighted_Price'].mean()



print("grouping by year") 
print(Yearly_Price.head(100)) 

Yearly_Price.plot(title="mean_weighted price per year", x="Year", y="mean price", legend="true") 
plt.show()  
print("plotting yearly price mean") 


df['Date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date
group = df.groupby('Date')
Daily_Price = group['Weighted_Price'].mean()
# descriptions of data 
print("describing data") 
print(Daily_Price)

prediction_days = 30
df_train= Daily_Price[:len(Daily_Price)-prediction_days]
df_test= Daily_Price[len(Daily_Price)-prediction_days:]

print("split")
print(len(df_train), len(df_test))


working_data = [df_train, df_test]
working_data = pd.concat(working_data)

working_data = working_data.reset_index()
working_data['Date'] = pd.to_datetime(working_data['Date'])
working_data = working_data.set_index('Date')

s = sm.tsa.seasonal_decompose(working_data.Weighted_Price.values, freq=60)
trace1 = go.Scatter(x = np.arange(0, len(s.trend), 1),y = s.trend,mode = 'lines',name = 'Trend',
    line = dict(color = ('rgb(244, 146, 65)'), width = 4))
trace2 = go.Scatter(x = np.arange(0, len(s.seasonal), 1),y = s.seasonal,mode = 'lines',name = 'Seasonal',
    line = dict(color = ('rgb(66, 244, 155)'), width = 2))

trace3 = go.Scatter(x = np.arange(0, len(s.resid), 1),y = s.resid,mode = 'lines',name = 'Residual',
    line = dict(color = ('rgb(209, 244, 66)'), width = 2))

trace4 = go.Scatter(x = np.arange(0, len(s.observed), 1),y = s.observed,mode = 'lines',name = 'Observed',
    line = dict(color = ('rgb(66, 134, 244)'), width = 2))

data = [trace1, trace2, trace3, trace4]
layout = dict(title = 'Seasonal decomposition', xaxis = dict(title = 'Time'), yaxis = dict(title = 'Price, USD'))
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='seasonal_decomposition')
py.plot(fig)
print("this is the graphhhhhhhh")


#
#
plt.figure(figsize=(15,7))
print("this is the graphhhhhhh2")
ax = plt.subplot(211)
print("this is the graphhhhstuck1")
sm.graphics.tsa.plot_acf(working_data.Weighted_Price.values.squeeze(), lags=48, ax=ax)
print("this is the graphhstuck2")
ax = plt.subplot(212)
print("this is the graphhhh3")
sm.graphics.tsa.plot_pacf(working_data.Weighted_Price.values.squeeze(), lags=48, ax=ax)
print("this is the graphhhh4")
plt.tight_layout()
print("this is the graphh5")
plt.show()
#
df_train = working_data[:-100]
df_test = working_data[-900:]
#
def create_lookback(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)
#
#
print("hello there 1")
training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1))
test_set = df_test.values
test_set = np.reshape(test_set, (len(test_set), 1))
#
##scale datasets
scaler = MinMaxScaler()
training_set = scaler.fit_transform(training_set)
test_set = scaler.transform(test_set)
#
## create datasets which are suitable for time series forecasting
look_back = 1
X_train, Y_train = create_lookback(training_set, look_back)
X_test, Y_test = create_lookback(test_set, look_back)
#
# # reshape datasets so that they will be ok for the requirements of the LSTM model in Keras
X_train = np.reshape(X_train, (len(X_train), 1, X_train.shape[1]))
X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))
#

model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(256))
model.add(Dense(1))  

model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, Y_train, epochs=300, batch_size=16, shuffle=False, 
                    validation_data=(X_test, Y_test), 
                    callbacks = [EarlyStopping(mode="auto", monitor='val_loss', min_delta=5e-5, patience=20, verbose=1)])
 
trace1 = go.Scatter(
    x = np.arange(0, len(history.history['loss']), 1),
    y = history.history['loss'],
    mode = 'lines',
    name = 'Train loss',
    line = dict(color=('rgb(66, 244, 155)'), width=2, dash='dash')
)
trace2 = go.Scatter(
    x = np.arange(0, len(history.history['val_loss']), 1),
    y = history.history['val_loss'],   
    mode = 'lines',
    name = 'Test loss',
    line = dict(color=('rgb(244, 146, 65)'), width=2)
)

data = [trace1, trace2]
layout = dict(title = 'Train and Test Loss during training', 
              xaxis = dict(title = 'Epoch number'), yaxis = dict(title = 'Loss'))
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='training_process')
py.plot(fig)  


# add one additional data point to align shapes of the predictions and true labels
X_test = np.append(X_test, scaler.transform(working_data.iloc[-1][0]))
X_test = np.reshape(X_test, (len(X_test), 1, 1))

# get predictions and then make some transformations to be able to calculate RMSE properly in USD
prediction = model.predict(X_test)
prediction_inverse = scaler.inverse_transform(prediction.reshape(-1, 1))
Y_test_inverse = scaler.inverse_transform(Y_test.reshape(-1, 1))
prediction2_inverse = np.array(prediction_inverse[:,0][1:])
Y_test2_inverse = np.array(Y_test_inverse[:,0])

trace1 = go.Scatter(
    x = np.arange(0, len(prediction2_inverse), 1),
    y = prediction2_inverse,
    mode = 'lines',
    name = 'Predicted price',
    hoverlabel= dict(namelength=-1),
    line = dict(color=('rgb(244, 146, 65)'), width=2)
)
trace2 = go.Scatter(
    x = np.arange(0, len(Y_test2_inverse), 1),
    y = Y_test2_inverse,
    mode = 'lines',
    name = 'True price',
    line = dict(color=('rgb(66, 244, 155)'), width=2)
)

data = [trace1, trace2]
layout = dict(title = 'Comparison of true prices (on the test dataset)hello with prices our model predicted',
             xaxis = dict(title = 'Day number'), yaxis = dict(title = 'Price, USD'))
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='results_demonstrating0')
py.plot(fig)

RMSE = sqrt(mean_squared_error(Y_test2_inverse, prediction2_inverse))
print('Test RMSE: %.3f' % RMSE)


Test_Dates = Daily_Price[len(Daily_Price)-1000:].index

trace1 = go.Scatter(x=Test_Dates, y=Y_test2_inverse, name= 'Actual Price', 
                   line = dict(color = ('rgb(66, 244, 155)'),width = 2))
trace2 = go.Scatter(x=Test_Dates, y=prediction2_inverse, name= 'Predicted Price',
                   line = dict(color = ('rgb(244, 146, 65)'),width = 2))
data = [trace1, trace2]
layout = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted, by dates',
             xaxis = dict(title = 'Date'), yaxis = dict(title = 'Price, USD'))
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='results_demonstrating1')
py.plot(fig)

 














# compile and fit the model
#model.compile(loss='mean_squared_error', optimizer='adam')
#history = model.fit(X_train, Y_train, epochs=100, batch_size=16, shuffle=False, 
#                    validation_data=(X_test, Y_test), 
#                    callbacks = [EarlyStopping(monitor='val_loss', min_delta=5e-5, patience=20, verbose=1)])

#array = df.values
#X = array[:,0:4] 
#print("errror 1")
#Y = array[:,4]
#print("errror 2") 
#Y=Y.astype('int')
#validation_size = 0.10
#seed = 7
#X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#print("errror 3") 

# Test options and evaluation metric
#seed = 7
#scoring = 'accuracy'

#model = Sequential()
#model.add(Dense(12, input_dim=4, activation='relu'))
#model.add(Dense(4, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
#
## Compile model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
## Fit the model
#model.fit(X, Y, epochs=150, batch_size=10)
#
## calculate predictions
#predictions = model.predict(X)
## round predictions
#rounded = [round(x[0]) for x in predictions]
#print(rounded)



# Spot Check Algorithms
#models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('DL', Sequential())) 
##models.append(('SVM', SVC()))
#print("errror 5") 
## evaluate each model in turn
#results = []
#names = []

#for name, model in models: 
#	kfold = model_selection.KFold(n_splits=10, random_state=seed)
#	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#	results.append(cv_results)
#	names.append(name)
#	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#	print(msg)
#print("errror 4")   
#fig = plt.figure()
#fig.suptitle('Algorithm Comparison')
#ax = fig.add_subplot(111)
#plt.boxplot(results)
#ax.set_xticklabels(names)
#plt.show()
#
#
#
## Make predictions on validation dataset
#knn = KNeighborsClassifier()
#knn.fit(X_train, Y_train)
#predictions = knn.predict(X_validation)
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))







#scatter_matrix(df)
#plt.show()
#
#
## histograms
#df.hist()
#plt.show()
