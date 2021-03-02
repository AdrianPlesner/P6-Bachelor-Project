import numpy as np
import properscoring as ps


def evaluate_forecast(df, forecast, length):
    observation = df.values.reshape(-1)[-length:]
    mean = forecast.mean
    std = np.std(forecast.samples, axis=0)
    e = ps.crps_gaussian(observation, mean, std)
    return np.average(e)
