from pathlib import Path
import data_processing as dp
import evaluation as evaluation
import tsv_to_gluon as tg
import h5_to_gluonts as hg
import make_forecast as fc
import random
import numpy as np
import sys
import json
import os
import time
from evaluation import Forecast

if __name__ == '__main__':
    if len(sys.argv) < 2:
        path = input("Missing program argument: metadata path\n"
                     "please give it:")
        if path == "":
            exit()
    with open(path) as md_file:
        md = json.load(md_file)
    print(str(md))
    print("loading data...")
    train, valid, test = hg.load_h5_to_gluon(md['path'], md)
    if md['normalize']:
        for data in (train, valid, test):
            for n in range(len(data)):
                data.list_data[n]['target'], data.list_data[n]['scaler'] = dp.preprocess_data(
                    data.list_data[n]['target'])
    predictor = fc.load_predictor(md['serialize_path'], md)
    test_slices = evaluation.split_validation(test, md)
    print("making predictions...")
    forecast = fc.make_forecast_vector(predictor, test_slices, md)
    if md['estimator'] == "TempFlow":
        forecast = [
            Forecast([slice[0].samples[::, ::, n] for n in range(325)], [slice[0].mean[::, n] for n in range(325)]) for
            slice in forecast]
    else:
        forecast = [Forecast([sensor.samples for sensor in slice], [sensor.mean for sensor in slice]) for slice in
                    forecast]
    print("Rescaling...")
    test_slices, forecast = dp.postprocess_data_vector(test_slices, forecast)
    slices = dp.listdata_to_array(test_slices)
    print("evaluating...")
    evals = np.stack(evaluation.validate_mp(slices[1:], forecast[:len(forecast) - 1]))
    e = np.average(evals)
    print(f"evaluation on test is {e}")
    exit()
