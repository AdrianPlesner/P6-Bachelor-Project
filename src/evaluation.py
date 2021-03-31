import numpy as np
import properscoring as ps
import pandas as pd
from gluonts.dataset.common import ListDataset


def evaluate_forecast(data, forecast, length):
    observation = data.list_data[0]['target'].reshape(-1, 1)[-length:]
    mean = forecast.mean
    std = np.std(forecast.samples, axis=0)
    e = ps.crps_gaussian(observation, mean, std)
    return np.average(e)


def split_validation(data, md):
    step = md['prediction_length']
    t = pd.date_range(start=data.list_data[0]['start'], freq=md['freq'], periods=len(data.list_data[0]['target']))
    return [ListDataset([{
        'start': t[n],
        'target': d['target'][n:n + step],
        'sensor_id': d['sensor_id'][n:n + step],
        'time_feat': d['time_feat'][::, n:n + step],
        'scaler': d['scaler']
    } for d in data.list_data], freq=md['freq']) for n in range(0, len(t), step)]


def validate(data, forecast):
    x = [np.sort(n.samples, 0) for n in forecast]
    evaluation = []
    for n in range(1):
        ar = x[n].swapaxes(0, 1)
        cdf = [CdfShell(a) for a in ar]
        xmin, xmax = [-np.inf for num in ar], [np.inf for num in ar]
        evaluation.append(ps.crps_quadrature(data.list_data[n]['target'], cdf))
    return evaluation


class CdfShell:
    def __init__(self, a):
        self.x = a
        self.y = np.arange(len(a)) / float(len(a))

    x = []
    y = []

    def cdf(self, a):
        v = np.searchsorted(self.x, a, 'left')
        if v == len(self.y):
            return 1.0
        return self.y[v]
