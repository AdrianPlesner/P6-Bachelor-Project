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


def postprocess_data(ld, forecast):

    for n in range(len(ld.list_data)):
        ld.list_data[n]['target'] = rescale_data(ld.list_data[n]['target'], ld.list_data[n]['scaler'])
        f_mean = rescale_data(forecast[n].mean.reshape(-1, 1), ld.list_data[n]['scaler']).reshape(-1)
        for m in range(len(f_mean)):
            forecast[n].mean[m] = f_mean[m]
        f_median = rescale_data(forecast[n].median.reshape(-1, 1), ld.list_data[n]['scaler']).reshape(-1)
        for m in range(len(f_median)):
            forecast[n].median[m] = f_median[m]
        for m in range(len(forecast[n].samples)):
            sample = rescale_data(forecast[n].samples[m].reshape(-1, 1), ld.list_data[n]['scaler']).reshape(-1)
            for k in range(len(sample)):
                forecast[n].samples[m][k] = sample[k]
    return ld, forecast


postprocess_data_vector = np.vectorize(postprocess_data, otypes=[list, list])


def make_prediction_interval(x, p):
    std = np.std(x.samples, axis=0)
    top = x.median + p * std
    bottom = x.median - p * std
    return top, bottom
