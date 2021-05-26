

import pandas as pd
import math
import conf
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler

pd.options.mode.chained_assignment = None

from conf import filepath, offset_prediction, known_time_periods

scaler = MinMaxScaler()


def getData(index):
    """
    Retrieve the data from a .5h file
    param index: The index of the sensor to get data form
    return: training set, testing set and validation set
    """
    # Get the data from the file
    file = pd.HDFStore(filepath, mode='r')
    test = file.get('test').iloc[:, [index]]
    train = file.get('train').iloc[:, [index]]
    validation = file.get('validation').iloc[:, [index]]
    """ Rename Columns """
    test.rename(columns={test.columns[0]: "Speed"}, inplace=True)
    train.rename(columns={train.columns[0]: "Speed"}, inplace=True)
    validation.rename(columns={validation.columns[0]: "Speed"}, inplace=True)
    """ Set the Time set frequences """
    test = test.asfreq('5min')
    train = train.asfreq('5min')
    validation = validation.asfreq('5min')
    """ Create additional variables  """
    test['Weekday'] = test.index.weekday
    test['Time'] = test.index.hour + test.index.minute / 60
    train['Weekday'] = train.index.weekday
    train['Time'] = train.index.hour + train.index.minute / 60
    validation['Weekday'] = validation.index.weekday
    validation['Time'] = validation.index.hour + validation.index.minute / 60
    file.close()
    return train, test, validation


def set_normalizer(data):
    """
    Set the base for the normalization
    param data: The data set to use for normalization
    """
    scaler.fit(data.loc[:, ['Speed']])
    return


def normalizeData(data):
    """
    Normalize a data set. The set_normalizer function needs to have been called before use
    param data: The data set to normalize
    return: Normalized data
    """
    # print('Normalize')
    data = scaler.transform(data.to_frame(name='Speed'))
    return data


def denormlizeData(data):
    """
    Denormalize a data set. The set_normalizer function needs to have been called before use.
    param data: The data set to denormalize, should have been normalized before use
    return: Denormalized data
    """
    if isinstance(data, np.ndarray):
        data = scaler.inverse_transform(data)
    else:
        data = scaler.inverse_transform(data.to_frame(name='Speed'))
    return data


def handleNanValuesSetAsSecondBefore(data):
    """
    Handle nan values in a data set
    param data: the working data set
    return: return a data set with out nan values
    """
    data = data.fillna(method='pad')
    return data


def handleZeroValuesSetAsSecondBefore(data):
    """
    Handle zero values in a data set
    param data: the working data set
    return: return a data set with out nan values
    """
    while (len(data[data['Speed'] == 0]) > 0):
        data[data['Speed'] == 0] = data.shift(-2)

    return data




def get_generator(data):
    """
    Convert a data set into a generator, which can be used in a LSTM model
    param data: the data set to convert
    return: data converted to a time series generator
    """
    generator = TimeseriesGenerator(data[:-offset_prediction], data[offset_prediction:].loc[:, 'Speed'],
                                    length=known_time_periods,
                                    batch_size=conf.batchsize)
    return generator


# Set all zero values to the mean of the collected speed before and the first non zero speed after
def handleZeroValuesFirstLastMean(data):
    """
    Handle zero values in a data set, by using the mean
    param data: the working data set
    return: return a data set with out nan values
    """
    for i in range(len(data)):
        if (data.iloc[i].loc["Speed"] == 0):
            preValue = data.iloc[i - 1].loc["Speed"]
            nextValue = 0
            for j in range(i, len(data)):
                if (data.iloc[j].loc["Speed"] != 0):
                    nextValue = data.iloc[j].loc["Speed"]
                    break

            # print('Pre Value: ' + str(preValue) + ' Next Value: ' + str(nextValue))
            data.iloc[i].loc["Speed"] = (preValue + nextValue) / 2
    return data


# Add a minimum and maximum off set to the prediction table.
# Where we take the mean of the average of off the model usually is a the current time of day of the week.
def meanProperbility(train, test):
    # Should be optimized
    for quentileset in conf.quantile:
        test[quentileset[0]] = 0
    train['Off'] = train['Training_Prediction'] - train['Speed']
    for index, row in test.iterrows():
        AtDateTime = (train.index.time == row.name.time()) & (train.index.weekday == row.name.weekday())
        for quentileset in conf.quantile:
            test.at[index, quentileset[0]] = getQuentaile(train[AtDateTime]['Off'], quentileset[1])

    for quentileset in conf.quantile:
        test[quentileset[0]] = test['Training_Prediction'] - test[quentileset[0]]
    return test


def getQuentaile(array, quentile):
    if quentile == 0:
        return np.amin(array)
    if quentile == 1:
        return np.amax(array)

    return np.quantile(array, quentile)
