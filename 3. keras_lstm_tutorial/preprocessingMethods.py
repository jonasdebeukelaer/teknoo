import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

def date_parser(date_string):
	return datetime.strptime(date_string, "%Y-%m")

def getAndFeatureEngineer():
	print('load data and feature engineer...')
	df = pd.read_csv('data/international_airline_passengers.csv', parse_dates=[0], date_parser=date_parser, engine='python', skipfooter=3)
	#df = pd.read_csv('data/international_airline_passengers.csv', engine='python', skipfooter=3)
	df['month_number'] = df['Month'].dt.month
	df = df.drop(['Month'], axis=1)
	dataset = df.values.astype('float32')

	scaler = MinMaxScaler(feature_range=(0,1))

	return dataset, scaler


def rescaleDataset(dataset, scaler):

	print(dataset[0:10])

	passengers = dataset[:, 0]
	month = dataset[:, 1]
	passengers = scaler.fit_transform(passengers)

	monthScaler = MinMaxScaler(feature_range=(0,1))
	month = monthScaler.fit_transform(month)


	dataset = np.array([passengers, month]).transpose()

	print("converted")
	print(dataset[0:10])

	return dataset

def splitDataset(dataset, look_back):
	train_size = int(len(dataset) * (2/3))
	test_size = len(dataset) - train_size + look_back
	train, test = dataset[0:train_size,:], dataset[train_size-look_back:,:]
	print(len(train), len(test))
	return train, test

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset) - look_back - 1):
		a = dataset[i:(i+look_back), 0]
		#b = dataset[i:(i+look_back), 1]
		#dataX.append([a, b])
		dataX.append([a])
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def preprocessDataset(dataset, look_back, scaler):
	print('preprocess...')

	dataset = rescaleDataset(dataset, scaler)
	(train, test) = splitDataset(dataset, look_back)

	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)

	trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[2], trainX.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], testX.shape[2], testX.shape[1]))

	return (trainX, trainY), (testX, testY)
