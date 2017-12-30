
import numpy as np

def rescaleDataset(dataset, scaler):
	dataset = scaler.fit_transform(dataset)
	return dataset

def splitDataset(dataset):
	train_size = int(len(dataset) * (2/3))
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:,:]
	print(len(train), len(test))
	return train, test

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset) - look_back - 1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def preprocessDataset(dataset, look_back, scaler):
	dataset = rescaleDataset(dataset, scaler)
	(train, test) = splitDataset(dataset)

	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)

	trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]), -1)
	testX = np.reshape(testX, (testX.shape[0], testX.shape[1]), -1)

	trainX = np.expand_dims(trainX, axis=3)
	testX = np.expand_dims(testX, axis=3)

	return (trainX, trainY), (testX, testY)