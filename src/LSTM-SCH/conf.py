import pathlib

""" Main Variables to set """
# The location of the data file, which should be fitted
# File Paths '/data/pems-bay.h5' or '/dataset/metr-la.h5'
filepath = str(pathlib.Path(__file__).parent.parent.absolute()) + '/dataset/pems-bay.h5'
# Defines the set of sensors which should be part of the batch. {start_sensor:end_sensor{
start_sensor = 0
end_sensor = 325  # Excluded
# Number of max threads
max_threads = 5

""" Model settings """
# Number of training runs
epochs = 100
# The number of batches the training set should be split into.
batchsize = 150
# LSTM number of nodes
lstm_nodes = 200
# How fare into the future should the model try and predict.
offset_prediction = 12
# How many time stamps does the model know, before making a prediction
known_time_periods = 12
# How many input variables does the set have.
n_input = 3
# Quantile Regression
quantile = [['min', 0], ['Q1', 0.25], ['Q2', 0.5], ['Q3', 0.75], ['max', 1]]
# Should it use true value for all test predictions or use rolling predictions for each new prediction
should_use_rolling = False
# If the run should be done on the testing set or the validation set
isValidating = False


def getTotalOffset():
    return known_time_periods + offset_prediction
