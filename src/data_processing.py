from sklearn import preprocessing
import numpy as np
from evaluation import DataSlice


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
        for m in range(len(forecast.samples[n])):
            sample = rescale_data(forecast.samples[n][m].reshape(-1, 1), ld.list_data[n]['scaler']).reshape(-1)
            for k in range(len(sample)):
                forecast.samples[n][m][k] = sample[k]
        mean = rescale_data(forecast.mean[n], ld.list_data[n]['scaler'])
        forecast.mean[n] = mean

    return ld, forecast


postprocess_data_vector = np.vectorize(postprocess_data, otypes=[list, list])


def make_prediction_interval(samples, p):
    s = np.sort(samples, axis=0)
    t = (p + ((100 - p)/2))/100
    b = 1-t
    t = t*(len(samples) + 1)
    b = b * (len(samples) + 1)
    top = s[int(t)]
    bottom = s[int(b)]
    return top, bottom


def listdata_to_array(data):
    result = []
    for n in data:
        result.append(DataSlice([m['target'] for m in n.list_data]))
    return np.asarray(result)
