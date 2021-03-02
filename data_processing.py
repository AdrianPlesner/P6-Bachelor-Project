from sklearn import preprocessing
import numpy as np


def preprocess_data(x):
    arr = np.array(x.reshape(-1, 1))
    scaler = preprocessing.StandardScaler().fit(arr)
    return scaler.transform(arr)


def preprocess_data_set(x):
    result = []
    for n in range(len(x)):
        result.append(preprocess_data(x[n]))
    return result
