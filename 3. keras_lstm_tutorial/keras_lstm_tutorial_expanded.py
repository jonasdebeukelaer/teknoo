# following https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

import matplotlib.pyplot as plt

import numpy as np
np.random.seed(5)

import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error

from preprocessingMethods import preprocessDataset, getAndFeatureEngineer

look_back = 3
batch_size = 1

dataset, scaler = getAndFeatureEngineer()

(trainX, trainY), (testX, testY) = preprocessDataset(dataset, look_back, scaler)
features = trainX.shape[2]

def defineModel():
	model = Sequential()
	model.add(LSTM(4, batch_input_shape=(batch_size, look_back, features), stateful=False, return_sequences=True))
	model.add(LSTM(4, batch_input_shape=(batch_size, look_back, features), stateful=False))
	model.add(Dense(1))

	model.compile(loss='mean_squared_error', optimizer='adam')

	return model

model = defineModel()

print('fit model...')
for i in range(1):
	print("loop {}".format(i), end=" ")
	model.fit(trainX, trainY, epochs=500, batch_size=batch_size, verbose=2, shuffle=True)
	#model.reset_states()

trainPredict = model.predict(trainX, batch_size=batch_size)
testPredict = model.predict(testX, batch_size=batch_size)

#rescale
trainPredict = scaler.inverse_transform(trainPredict)
testPredict = scaler.inverse_transform(testPredict)
trainY = scaler.inverse_transform(trainY)
testY = scaler.inverse_transform(testY)

trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:, 0]))
print('Train Score: {:02.2f}'.format(trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict[:, 0]))
print('Test Score: {:02.2f}'.format(testScore))

def plotThings(dataset, trainPredict, testPredict):
	trainPredictPlot = np.empty_like(dataset)
	trainPredictPlot[:,:] = np.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

	testPredictPlot = np.empty_like(dataset)
	testPredictPlot[:,:] = np.nan
	testPredictPlot[len(trainPredict)+look_back+1:len(dataset)-1, :] = testPredict

	plt.plot(dataset[:, 0])
	plt.plot(trainPredictPlot)
	plt.plot(testPredictPlot)
	plt.show()

plotThings(dataset, trainPredict, testPredict)



