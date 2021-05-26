import numpy as np
import time
import scipy.integrate as integrate


def createPredictionArrays(train, test):
    """
    Create a list of sets which contain a set of prediction values for each prediction
    param train: the training set containing training predictions
    param test: the testing data with also contains predictions
    return: a list of prediction sets for each point prediction in the testing set
    """
    returnArray = []
    for index, row in test.iterrows():
        predictionArray = row['Training_Prediction'] - train[(train.index.time == row.name.time()) & (train.index.weekday == row.name.weekday())]['Off']
        predictionArray = np.sort(predictionArray)
        guasstion_array = np.random.normal(loc=predictionArray.mean(), scale=np.std(predictionArray), size=300)
        guasstion_array = np.sort(guasstion_array)
        returnArray.append(guasstion_array)
    return returnArray


def validate(train, test):
    """
    Validate the results of the LSTM model
    param train: the training set containing training predictions
    param test: the testing data with also contains predictions
    return: CRPS validation set for each prediction, mean square error value for each prediction
    """
    start = time.perf_counter()
    true_values = test['Speed']
    forecast = createPredictionArrays(train, test)
    # x = [np.sort(n.samples, 0) for n in forecast]
    evaluation = []
    SquareError = []
    for n in range(len(forecast)):
        ar = forecast[n]
        SquareError.append((ar.mean() - true_values[n]) ** 2)
        #cdf = [CdfShell(a) for a in ar]
        cdf = CdfShell(ar)
        b = crps_vector(true_values[n], cdf)
        evaluation.append(b)
    end = time.perf_counter() - start
    return np.asarray(evaluation), np.asarray(SquareError)



validate_vector = np.vectorize(validate, otypes=[list])


def _crps(val, a):
    x = a.x
    y = a.y
    split = np.searchsorted(x, val)
    if split < 0:
        split = 0
    if split == len(x):
        split -= 1
    lhs = np.square(y[:split])
    rhs = np.square(1 - y[split:])
    if len(lhs) == 0:
        lc = 0
    else:
        lc = integrate.trapezoid(lhs, x[:split])
    if len(rhs) == 0:
        rc = 0
    else:
        rc = integrate.trapezoid(rhs, x[split:])

    result = lc + rc
    if result < 0:
        stoping = 0
    return result


crps_vector = np.vectorize(_crps, otypes=[list])


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
