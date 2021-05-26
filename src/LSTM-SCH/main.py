from multiprocessing.context import Process
from multiprocessing import Queue
import xlsxwriter
import multiprocessing
import DataManager as DM
import modelmanager as MM
import numpy as np
import conf
import sys
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
import tensorflow as tf
import crps

# demand_training_data = trainingdata
results = {}


def LSTM_CHS(index):
    """
    Runs the LSTM model on the data from the a sensor and returns it's validation data
    :param index: The index of the sensor to run
    :return: Index of the sensor, the crps score and the mse score
    """
    # Get the data
    print("Running : " + index)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    train, test, validate = DM.getData(int(index))
    # Remove the 0 value data points
    train = DM.handleNanValuesSetAsSecondBefore(train)
    test = DM.handleNanValuesSetAsSecondBefore(test)
    validate = DM.handleNanValuesSetAsSecondBefore(validate)
    # Split the data into a training and test set
    prediction_set = validate if conf.isValidating else test
    # Normalize the data
    DM.set_normalizer(train)
    train_norm = train
    train_norm['Speed'] = DM.normalizeData(train['Speed'])
    prediction_norm = prediction_set
    prediction_norm['Speed'] = DM.normalizeData(prediction_set['Speed'])
    # scale the training set to a 3 dimensional shape
    generated_training = DM.get_generator(train_norm)
    generated_predictions = DM.get_generator(prediction_norm)

    # Get the model
    model = MM.get_basic_propalisic_model()
    # Train the model
    History = model.fit(generated_training, epochs=conf.epochs, verbose=2).history

    training_prediction = model.predict(generated_training)
    training_prediction = DM.denormlizeData(training_prediction)
    train['Speed'] = DM.denormlizeData(train['Speed'])
    train = train.iloc[conf.getTotalOffset():]
    train['Training_Prediction'] = training_prediction

    if conf.should_use_rolling:
        MM.rolling_prediction(model, test, 12 * 24)
    else:
        predicted_predictions = model.predict(generated_predictions)
        predicted_predictions = DM.denormlizeData(predicted_predictions)
        prediction_set = prediction_set.iloc[conf.getTotalOffset():]
        # test = test.iloc[conf.known_time_periods: -conf.offset_prediction]
        prediction_set['Training_Prediction'] = predicted_predictions

    prediction_set['Speed'] = DM.denormlizeData(prediction_set['Speed'])
    prediction_set = DM.meanProperbility(train, prediction_set)

    # Get the validation scores for CRPS and MSE
    validations_scores, SquareError = crps.validate(train, prediction_set)
    validations_score_mean = np.mean(validations_scores)
    SquareErrorMean = SquareError.mean()
    print(index + " Score : " + str(validations_score_mean) + "\n \t MSE: " + str(SquareErrorMean))
    # Write the line to fine
    return index, validations_score_mean, SquareErrorMean


if __name__ == '__main__':
    filename = sys.argv[0]
    children = []
    start = conf.start_sensor
    end = conf.end_sensor
    # Number of max threads
    count = conf.max_threads

    # Start running on the sensors
    pool = multiprocessing.Pool(processes=count)
    results = [pool.apply_async(LSTM_CHS, (str(i),)) for i in range(start,end)]
    results = [res.get() for res in results]
    print(f'result: {results}')

    # Create excel file
    workbook = xlsxwriter.Workbook('Expenses01.xlsx')
    worksheet = workbook.add_worksheet()

    for item in results:
        worksheet.write(int(item[0]), 0, item[1])
        worksheet.write(int(item[0]), 1, item[2])

    workbook.close()
    print(results)

