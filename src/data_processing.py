from sklearn import preprocessing
import numpy as np
from evaluation import DataSlice


def preprocess_data(x):
    """
    Scale data to have 0 mean and a standard deviation
    :param x: List/array of data to be scaled
    :return: tuple of transformed data and scaler for rescaling
    """
    arr = np.array(x).reshape(-1, 1)
    scaler = preprocessing.StandardScaler().fit(arr)
    return scaler.transform(arr).reshape(-1), scaler


def rescale_data(x, scaler):
    """
    Rescale data back to its orginal form from preprocess_data()
    :param x: List/array of data to rescale
    :param scaler: scaler object returned by preprocess_data()
    :return: Array of rescaled data
    """
    # x is a (n,1) column vector
    return scaler.inverse_transform(x)


def postprocess_data(ld, forecast):
    """
    Post process a whole ListData object and forecasts to have original scale
    :param ld: ListData object containing data and scalers
    :param forecast: Forecasts objects corresponding to the data in ld
    :return: ListData object and forecast list, rescale to original scale
    """
    for n in range(len(ld.list_data)):
        ld.list_data[n]['target'] = rescale_data(ld.list_data[n]['target'], ld.list_data[n]['scaler'])
        for m in range(len(forecast.samples[n])):
            sample = rescale_data(forecast.samples[n][m].reshape(-1, 1), ld.list_data[n]['scaler']).reshape(-1)
            for k in range(len(sample)):
                forecast.samples[n][m][k] = sample[k]
        mean = rescale_data(forecast.mean[n], ld.list_data[n]['scaler'])
        forecast.mean[n] = mean

    return ld, forecast


postprocess_data_vector = np.vectorize(postprocess_data, otypes=[list, list])
"""
Vectorized version of postprocess_data()
"""


def make_prediction_interval(samples, p):
    """
    Computes a prediction interval of size p based on samples
    :param samples: The samples for which to base the prediction interval
    :param p: The percentage size of the interval [0-100]
    :return: a tuple of arrays containing the top and bottom limits of the interval
    """
    s = np.sort(samples, axis=0)
    t = (p + ((100 - p)/2))/100
    b = 1-t
    t = t*(len(samples) + 1)
    b = b * (len(samples) + 1)
    top = s[int(t)]
    bottom = s[int(b)]
    return top, bottom


def listdata_to_array(data):
    """
    Transfrom a list of ListData objects to a list of DataSlice objects
    :param data: List of ListData objects
    :return: Array of DataSlice objects
    """
    result = []
    for n in data:
        result.append(DataSlice([m['target'] for m in n.list_data]))
    return np.asarray(result)
