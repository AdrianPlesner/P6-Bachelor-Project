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
    l = list()
    t = pd.date_range(start=data.list_data[0]['start'], freq=md['freq'], periods=len(data.list_data[0]['target']))
    l = [ListDataset([{

    }])]

    return l
