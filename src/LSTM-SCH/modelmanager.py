from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras.optimizers as optimizers
import numpy as np
import pandas as pd
import DataManager as DM

import conf



def get_basic_propalisic_model():
    """
    Returns the LSTM model
    return: the LSTM model
    """
    model = Sequential()
    model.add(LSTM(conf.lstm_nodes, activation='relu', recurrent_activation='sigmoid',dropout=0,
                   input_shape=(conf.known_time_periods, conf.n_input)))
    model.add(Dense(1))
    # model.add(tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)))
    model.compile(optimizer='adam', loss='mse')
    return model


def rolling_prediction(model, test, number_of_predictions):
    """
    Returns the LSTM model, which can be used to create rolling forecasts
    return: the LSTM model
    """
    test_norm = DM.normalizeData(test)
    # test_set = test[:conf.known_time_periods]['Speed'].to_numpy()
    test_set = test_norm[:conf.known_time_periods]
    test_set = test_set.reshape((1, conf.known_time_periods, 1))
    predictions = []

    for i in range(number_of_predictions):
        current_pred = model.predict(test_set)[0]

        # store prediction
        predictions.append(current_pred)

        test_set = np.append(test_set[:, 1:, :], [[current_pred]], axis=1)

    predictions = DM.denormlizeData(predictions)
    test['Training_Prediction'] = np.nan
    temp = pd.DataFrame(data=predictions, columns=['Training_Prediction'], index=range(len(predictions)))
    temp.set_index(pd.date_range(test.index[conf.known_time_periods], periods=number_of_predictions, freq='5T'),
                   inplace=True)
    test.update(temp)
    return test
