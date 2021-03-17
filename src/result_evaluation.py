from pathlib import Path
import data_processing as dp
import evaluation as evaluation
import tsv_to_gluon as tg
import h5_to_gluonts as hg
import make_forecast as fc
import random
import csv_to_gluon as cg
import sys
import json
import os
from gluonts.model.predictor import Predictor
import numpy as np

### pems-bay data
### 325 sensors
### 52116 time steps
### we skip the first 288

### Covid19 data
### 274 locations
### 413 time steps

with open("results/Pems/5data-7days-4hours-1200its/metadata.json") as md_file:
    md = json.load(md_file)
train_length = 288
max_offset = 50000-1-288-48
min_offset = 7*288  # train data
evals = np.zeros((md['iterations'], 100, 2))

for n in range(md['iterations']):
    print(str(n))
    os.makedirs(md['deserialize_path'] + "pictures/" + str(n))
    for i in range(100):
        ### make predicton and evaluation with the same sensor
        # get random offset
        offset = random.randint(min_offset, max_offset)
        # load data
        train, test = hg.load_h5_to_gluon("data/pems-bay.h5", train_length, md['test_length'], offset, md['freq'], sensor=n)
        #train, test = cg.load_csv_to_gluon("data/time_series_covid19_confirmed_global.csv", train_length, md['test_length'], md['freq'], offset, md['location'])
        # normalize data
        train.list_data[0]['target'], train.list_data[0]['scaler'] = dp.preprocess_data(train.list_data[0]['target'])
        test.list_data[0]['target'], test.list_data[0]['scaler'] = dp.preprocess_data(test.list_data[0]['target'])
        data = {'train': train, 'test': test}
        # load predictor
        predictor = Predictor.deserialize(Path(md['deserialize_path'] + "p" + str(n)))
        predictor.prediction_net.ctx = predictor.ctx
        # make forecast
        forecast = list(predictor.predict(train))[0]
        data, forecast = dp.postprocess_data(data, forecast)
        # make a plot
        fc.plot_forecast(data['test'], forecast, n, md, path="plot" + str(i))
        # evaluation
        e = evaluation.evaluate_forecast(data['test'], forecast, md['test_length'])
        evals[n, i, 0] = e
        ### make prediction and evaluation with a random sensor
        # get random offset
        offset = random.randint(min_offset, max_offset)
        # get a random sensor
        sensor = random.randint(0, 324)
        # load data
        train, test = hg.load_h5_to_gluon("data/pems-bay.h5", train_length, md['test_length'], offset, md['freq'],sensor=sensor)
        #train, test = cg.load_csv_to_gluon("data/time_series_covid19_confirmed_global.csv", train_length,md['test_length'], md['freq'], offset, sensor)
        # normalize data
        train.list_data[0]['target'], train.list_data[0]['scaler'] = dp.preprocess_data(train.list_data[0]['target'])
        test.list_data[0]['target'], test.list_data[0]['scaler'] = dp.preprocess_data(test.list_data[0]['target'])
        data = {'train': train, 'test': test}
        # make forecast
        forecast = list(predictor.predict(train))[0]
        data, forecast = dp.postprocess_data(data, forecast)
        # make a plot
        fc.plot_forecast(data['test'], forecast, n, md, path="plot" + str(i + 100), sensor=sensor)
        # evaluation
        e = evaluation.evaluate_forecast(data['test'], forecast, md['test_length'])
        evals[n, i, 1] = e
with open(md['deserialize_path']+"evaluation.txt", "w") as file:
    file.write(str(md))
    for n in range(md['iterations']):
        file.write("\nSensor " + str(n) + ":\n")
        for i in range(100):
            file.write(str(evals[n, i, 0]) + "\n")
        file.write("Average = " + str(np.average(evals[n, ::, 0])) + "\n\n")
        file.write("Random sensor:\n")
        for i in range(100):
            file.write(str(evals[n, i, 1]) + "\n")
        file.write("Average = " + str(np.average(evals[n, ::, 1])) + "\n\n")
        file.write("Total average for sensor = " + str(np.average(evals[n])) + "\n\n\n")
    file.write("Total average for model = " + str(np.average(evals)))
print("done")


