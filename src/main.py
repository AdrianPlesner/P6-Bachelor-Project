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
        param = 'stop'
        i = 0
    else:
        path = sys.argv[1]
        param = 'stop'
        i = 0
        if len(sys.argv) > 2:
            param = sys.argv[2]
            i = int(sys.argv[3])
            # step = int(sys.argv[3])
            # div = int(sys.argv[4])
            # if div > 0:
            #     step = step / div

    with open(path) as md_file:
        md = json.load(md_file)
    print(str(md))
    ### Load data
    start = time.perf_counter()
    train, valid, test = hg.load_h5_to_gluon(md['path'], md)
    # if md['make_plots']:
    # hg.plot_train_test(train, test)
    if md['normalize']:
        for data in (train, valid, test):
            for n in range(len(data)):
                data.list_data[n]['target'], data.list_data[n]['scaler'] = dp.preprocess_data(
                    data.list_data[n]['target'])
    end = time.perf_counter() - start
    print("Loading data took", end, "seconds")
    # if md['make_plots']:
    # hg.plot_train_test(train, test)
    if 'params' in md:
        param = md['params'][i]
        md[param] = md['start'][i]
        step = md['step'][i]
        end = md['end'][i]
        best = 10.0
        result = 0
        for p in range(md[param], end, step):
            md[param] = p
            try:
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
                    predictor = fc.load_predictor(md['serialize_path'], md)

                ### Compute validation metrics
                validation_slices = evaluation.split_validation(valid, md)
                #validation_slices = validation_slices[:5]
                start = time.perf_counter()
                forecast = fc.make_forecast_vector(predictor, validation_slices, md)
                if md['estimator'] == "TempFlow":
                    forecast = [Forecast([slice[0].samples[::, ::, n] for n in range(325)], [slice[0].mean[::, n] for n in range(325)]) for slice in forecast]
                else:
                    forecast = [Forecast([sensor.samples for sensor in slice], [sensor.mean for sensor in slice]) for slice in forecast]
                end = time.perf_counter() - start
                print("Creating forecasts took", end, "seconds\n Start rescaling")
                start = time.perf_counter()
                validation_slices, forecast = dp.postprocess_data_vector(validation_slices, forecast)
                end = time.perf_counter() - start
                print("Rescaling took", end, "seconds\n Start validating...")
                start = time.perf_counter()
                slices = dp.listdata_to_array(validation_slices)
                evals = np.stack(evaluation.validate_mp(slices[1:], forecast[:len(forecast)-1]))
                end = time.perf_counter() - start
                print("Evaluation took", end, "seconds")
                e = np.average(evals)
                print(f"Parameter {param} with value {md[param]} had evaluation {e}")
                if e < best:
                    best = e
                    result = md[param]
            except:
                raise
        md[param] = result
        print(f"The best value for {param} is {result} with evaluation {best}")
        if not param == 'stop':
            print(f"The final result is {str(md)}\n with evaluation {best}")
            with open(path, "w") as jsonfile:
                json.dump(md, jsonfile)
            # with open(md['serialize_path'] + "evaluation.txt", "w") as file:
            #     e = np.stack(evals)
            #     for i in range(len(evals)):
            #         file.write("Slice: " + str(i) + "\n")
            #         for j in range(len(evals[i])):
            #             file.write("Sensor: " + str(j) + "\n")
            #             for k in range(len(evals[i][j])):
            #                 file.write("{:.2f}".format(evals[i][j][k]) + ", ")
            #             file.write("\n")
            #
            #         file.write("\n\n")
            #     file.write("Average: " + str(np.average(e)))
    else:
        ### Train network
        if md['train_predictor']:
            predictor = fc.train_predictor(train, md)
            if not os.path.isdir(md['serialize_path']):
                os.makedirs(md['serialize_path'])
            predictor.serialize(Path(md['serialize_path'] + "/"))
        else:
            ### Load pre-trained predictors
            predictor = fc.load_predictor(md['serialize_path'], md)

        ### Compute validation metrics
        validation_slices = evaluation.split_validation(valid, md)
        for _ in range(5):
            i = random.randint(1, len(validation_slices)-1)
            v_slices = validation_slices[i-1:i+1]
            forecast = fc.make_forecast_vector(predictor, v_slices, md)
            if md['estimator'] == "TempFlow":
                forecast = [Forecast([slice[0].samples[::, ::, n] for n in range(325)], [slice[0].mean[::, n] for n in range(325)]) for slice in forecast]
            else:
                forecast = [Forecast([sensor.samples for sensor in slice], [sensor.mean for sensor in slice]) for slice in forecast]
            v_slices, forecast = dp.postprocess_data_vector(v_slices, forecast)
            slices = dp.listdata_to_array(v_slices)
            evals = np.stack(evaluation.validate_mp(slices[1:], forecast[:len(forecast)-1]))
            e = np.average(evals)
            u = random.randint(0, 324)
            fc.plot_forecast(v_slices[0], v_slices[1], forecast[0], u, md, np.average(evals[0, u, ::]))
