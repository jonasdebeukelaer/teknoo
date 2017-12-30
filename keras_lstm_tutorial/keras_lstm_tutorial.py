# following https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
np.random.seed(5)

import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from preprocessingMethods import preprocessDataset

df = pd.read_csv('../data/international_airline_passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = df.values.astype('float32')

scaler = MinMaxScaler(feature_range=(0,1))
look_back = 3
(trainX, trainY), (testX, testY) = preprocessDataset(dataset, look_back, scaler)

batch_size = 1

def defineModel():
	timesteps = look_back
	features = 1

	model = Sequential()
	model.add(LSTM(4, batch_input_shape=(batch_size, timesteps, features), stateful=True, return_sequences=True))
	model.add(LSTM(4, batch_input_shape=(batch_size, timesteps, features), stateful=True))
	model.add(Dense(1))

	model.compile(loss='mean_squared_error', optimizer='adam')

	return model

model = defineModel()

for i in range(100):
	model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2, shuffle=False)
	model.reset_states()

trainPredict = model.predict(trainX, batch_size=batch_size)
testPredict = model.predict(testX, batch_size=batch_size)

trainPredict = scaler.inverse_transform(trainPredict)
trainY= scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY= scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: {:02.2f}'.format(trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: {:02.2f}'.format(testScore))

trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()