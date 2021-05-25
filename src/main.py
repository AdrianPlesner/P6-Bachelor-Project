from pathlib import Path
import data_processing as dp
import evaluation as evaluation
import h5_to_gluonts as hg
import make_forecast as fc
import numpy as np
import sys
import json
import os
import time
from evaluation import Forecast


def get_metadata():
    """
    Method for retrieving metadata dictionary. Metadata should be a json dictionary. Path is given as program argument
    or through console prompt
    :return: returns a python dictionary containing the metadata
    """
    if len(sys.argv) < 2:
        # The path is not given as program argument, prompt the user
        path = input("Missing program argument: metadata path\n"
                     "please give it:")
        if path == "":
            # No path is given. Cannot continue
            exit()
    else:
        # Path should be in the program argument
        path = sys.argv[1]
    # Read json file containing metadata
    with open(path) as md_file:
        md = json.load(md_file)
    print(str(md))
    return md


def load_data(md):
    """
    Loads data from .h5 to GluonTS ListData
    :param md: the metadata dictionary containing at least the entries 'path' : path to data and 'freq': frequence of
    data
    :return: a tuple of the training, validation and testing partitions of the data according to the metadata
    """
    train, valid, test = hg.load_h5_to_gluon(md['path'], md)

    if md['normalize']:
        # Normalize data if implied by metadata
        for data in (train, valid, test):
            for n in range(len(data)):
                data.list_data[n]['target'], data.list_data[n]['scaler'] = dp.preprocess_data(
                    data.list_data[n]['target'])
    return train, valid, test


def validate_predictor(data, predictor, md):
    """
    Validates CRPS for a given predictor and data
    :param data: Data to validate against
    :param predictor: Predictor to validate
    :param md: Metadata dictionary
    :return: The average CRPS evaluation of predictor against data
    """
    # Split data to 1 hour slices
    validation_slices = evaluation.split_validation(data, md)
    start = time.perf_counter()
    # Create forecasts
    forecast = fc.make_forecast_vector(predictor, validation_slices, md)
    if md['estimator'] == "TempFlow":
        forecast = [Forecast([slice[0].samples[::, ::, n] for n in range(md['sensors'])],
                             [slice[0].mean[::, n] for n in range(md['sensors'])]) for slice in forecast]
    else:
        forecast = [Forecast([sensor.samples for sensor in slice], [sensor.mean for sensor in slice]) for
                    slice in forecast]
    end = time.perf_counter() - start
    print("Creating forecasts took", end, "seconds\n Start rescaling")
    if md['normalize']:
        start = time.perf_counter()
        # Rescale data and forecasts if normalized
        validation_slices, forecast = dp.postprocess_data_vector(validation_slices, forecast)
        end = time.perf_counter() - start
        print("Rescaling took", end, "seconds\n Start validating...")
    start = time.perf_counter()
    # Validate forecasts against data
    slices = dp.listdata_to_array(validation_slices)
    evals = np.stack(evaluation.validate_mp(slices[1:], forecast[:len(forecast) - 1]))
    end = time.perf_counter() - start
    print("Evaluation took", end, "seconds")
    e = np.average(evals)
    return e


def get_predictor(data, md):
    """
    Gets a predictor, either through training or de-serialization
    :param data: data to train on if training is necessary
    :param md: metadata dictionary containing at least 'train_predcitor' and 'serialize_path'
    :return: predictor
    """

    if md['train_predictor']:
        ### Train network
        start = time.perf_counter()
        predictor = fc.train_predictor(data, md)
        end = time.perf_counter() - start
        print("Training the predictor took", end, "seconds")
        if not os.path.isdir(md['serialize_path']):
            os.makedirs(md['serialize_path'])
        predictor.serialize(Path(md['serialize_path'] + "/"))
    else:
        ### Load pre-trained predictors
        predictor = fc.load_predictor(md['serialize_path'], md)
    return predictor


def iterate_params(train, valid, md):
    """
    Iterates over a set of values for a set of hyper parameters to learn the best combination
    :param train: The training set
    :param valid: The validation set
    :param md: Metadata dictionary containing parameters and their intervals
    :return:
    """
    for i in range(len(md['params'])):
        param = md['params'][i]
        s = md['start'][i]
        if param == 'distribution':
            dists = md[param]
            md[param] = dists[s]
        else:
            md[param] = s
        step = md['step'][i]
        end = md['end'][i]
        best = 10000.0
        result = 0
        for p in range(s, end, step):
            if param == "distribution":
                md[param] = dists[p]
            else:
                md[param] = p
                predictor = get_predictor(train, md)

                ### Compute validation metrics
                e = validate_predictor(valid, predictor, md)
                print(f"Parameter {param} with value {md[param]} had evaluation {e}")
                if e < best:
                    best = e
                    result = md[param]
        md[param] = result
        print(f"The best value for {param} is {result} with evaluation {best}")
        if not param == 'stop':
            print(f"The final result is {str(md)}\n with evaluation {best}")
            with open(md['path'], "w") as jsonfile:
                json.dump(md, jsonfile)
    return md


if __name__ == '__main__':
    md = get_metadata()
    ### Load data
    start = time.perf_counter()
    train, valid, test = load_data(md)
    end = time.perf_counter() - start
    print("Loading data took", end, "seconds")

    if 'params' in md:
        iterate_params(train, valid, md)
    else:
        predictor = get_predictor(train, md)
        e = validate_predictor(valid, predictor, md)
        print(f"Model has validation CRPS: {e}")
