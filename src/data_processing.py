from sklearn import preprocessing
import numpy as np


def preprocess_data(x):
    arr = np.array(x).reshape(-1, 1)
    scaler = preprocessing.StandardScaler().fit(arr)
    return scaler.transform(arr).reshape(-1), scaler


def preprocess_data_set(x):
    result = []
    for n in range(len(x)):
        result.append(preprocess_data(x[n]))
    return result


def rescale_data(x, scaler):
    # x is a (n,1) column vector
    return scaler.inverse_transform(x)


def postprocess_data(test, forecast, scaler):
    test_values = rescale_data(test.values, scaler)
    for n in range(len(test_values)):
        test.values[n] = test_values[n]
    f_mean = rescale_data(forecast.mean.reshape(-1, 1), scaler).reshape(-1)
    for n in range(len(f_mean)):
        forecast.mean[n] = f_mean[n]
    f_median= rescale_data(forecast.median.reshape(-1, 1), scaler).reshape(-1)
    for n in range(len(f_median)):
        forecast.median[n] = f_median[n]
    for n in range(len(forecast.samples)):
        sample = rescale_data(forecast.samples[n].reshape(-1, 1), scaler).reshape(-1)
        for m in range(len(sample)):
            forecast.samples[n][m] = sample[m]
    return test, forecast


def make_prediction_interval(x, p):
    std = np.std(x, axis=0)
    top = x + p * std
    bottom = x - p * std
    return top, bottom
