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

if __name__ == '__main__':
    if len(sys.argv) < 2:
        path = input("Missing program argument: metadata path\n"
                     "please give it:")
        if path == "":
            exit()
    else:
        path = sys.argv[1]

    with open(path) as md_file:
        md = json.load(md_file)

    ### Load data
    start = time.perf_counter()
    train, valid, test = hg.load_h5_to_gluon(md['path'], md)
    # if md['make_plots']:
    # hg.plot_train_test(train, test)
    if md['normalize']:
        for data in (train, valid, test):
            for i in range(len(data)):
                data.list_data[i]['target'], data.list_data[i]['scaler'] = dp.preprocess_data(
                    data.list_data[i]['target'])
    end = time.perf_counter() - start
    print("Loading data took", end, "seconds")
    # if md['make_plots']:
    # hg.plot_train_test(train, test)

    ### Train network
    if md['train_predictor']:
        start = time.perf_counter()
        predictor = fc.train_predictor(train, md)
        end = time.perf_counter() - start
        print("Training the predictor took", end, "seconds")
        if not os.path.isdir(md['serialize_path']):
            os.makedirs(md['serialize_path'])
        predictor.serialize(Path(md['serialize_path'] + "/"))
    else:
        ### Load pre-trained predictors
        predictor = fc.load_predictor(md['serialize_path'])

    ### Compute validation metrics
    validation_slices = evaluation.split_validation(valid, md)
#    validation_slices = validation_slices
    start = time.perf_counter()
    forecast = fc.make_forecast_vector(predictor, validation_slices)
    end = time.perf_counter() - start
    print("Creating forecasts took", end, "seconds")
    start = time.perf_counter()
    validation_slices, forecast = dp.postprocess_data_vector(validation_slices, forecast)
    end = time.perf_counter() - start
    print("Rescaling took", end, "seconds")
    start = time.perf_counter()
    evals = evaluation.validate_mp(validation_slices, forecast)
    end = time.perf_counter() - start
    print("Evaluation took", end, "seconds")

    with open(md['serialize_path'] + "evaluation.txt", "w") as file:
        e = np.stack(evals)
        for i in range(len(evals)):
            file.write("Slice: " + str(i) + "\n")
            for j in range(len(evals[i])):
                file.write("Sensor: " + str(j) + "\n")
                for k in range(len(evals[i][j])):
                    file.write(str(evals[i][j][k]) + ", ")
                file.write("\n")

            file.write("\n\n")
        file.write("Average: " + str(np.average(e)))
