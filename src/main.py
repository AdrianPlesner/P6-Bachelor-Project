from pathlib import Path
import data_processing as dp
import evaluation as evaluation
import tsv_to_gluon as tg
import h5_to_gluonts as hg
import make_forecast as fc
import random

import sys
import json
import os


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

train, valid, test = hg.load_h5_to_gluon(md['path'], md)
#if md['make_plots']:
    #hg.plot_train_test(train, test)
if md['normalize']:
    for data in (train, valid, test):
        for n in range(len(data)):
            data.list_data[n]['target'], data.list_data[n]['scaler'] = dp.preprocess_data(data.list_data[n]['target'])

#if md['make_plots']:
    #hg.plot_train_test(train, test)

### Train network
if md['train_predictor']:
    predictor = fc.train_predictor(train, md)
    if not os.path.isdir(md['serialize_path']):
        os.makedirs(md['serialize_path'])
    predictor.serialize(Path(md['serialize_path'] + "/"))
else:
    ### Load pre-trained predictors
    predictor = fc.load_predictor(md['serialize_path'])

### Compute validation metrics


# ### Make forecasts
# forecast = fc.make_forecast(predictor, data, md)
# if md['normalize']:
#     for n in range(iterations):
#         data[n], forecast[n] = dp.postprocess_data(data[n], forecast[n])
#
# ### Plot forecasts
# if md['make_plots']:
#     for n in range(iterations):
#         fc.plot_forecast(data[n]['test'], forecast[n], n, md)
#
# ### Evaluate predictions
# evals = []
# for n in range(iterations):
#     e = evaluation.evaluate_forecast(data[n]['test'], forecast[n], md['test_length'])
#     evals.append(e)
# print(evals)
