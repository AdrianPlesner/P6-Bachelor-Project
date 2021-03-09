import numpy as np
import properscoring as ps


def evaluate_forecast(data, forecast, length):
    observation = data.list_data[0]['target'].reshape(-1, 1)[-length:]
    median = forecast.median
    std = np.std(forecast.samples, axis=0)
    e = ps.crps_gaussian(observation, median, std)
    return np.average(e)
