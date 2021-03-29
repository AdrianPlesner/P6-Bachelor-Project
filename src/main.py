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


if len(sys.argv) < 2:
    path = input("Missing program argument: metadata path\n"
                 "please give it:")
    if path == "":
        exit()
else:
    path = sys.argv[1]

with open(path) as md_file:
    md = json.load(md_file)

data = []
iterations = md['iterations']
offset = md['offset']  # random.randint(0, 7*288 - 12 * 24)


### Load data
for n in range(iterations):
    if md['data_type'] == 'csv':
        train, test = cg.load_csv_to_gluon(md['path'], md['train_length'], md['test_length'], md['freq'], offset,
                                           md['location'], md['sum'])
    elif md['data_type'] == 'tsv':
        train, test = tg.load_tsv_to_gluon(md['path'], md['train_length'], md['test_length'], n, md['freq'])
    elif md['data_type'] == 'h5':
        train, test = hg.load_h5_to_gluon(md['path'], md['train_length'], md['test_length'], offset, md['freq'],
                                          md['h5_key'], n)
    else:
        print("No data type in metadata")
        exit()
    #if md['make_plots']:
        #hg.plot_train_test(train, test)
    if md['normalize']:
        train.list_data[0]['target'], train.list_data[0]['scaler'] = dp.preprocess_data(train.list_data[0]['target'])
        test.list_data[0]['target'], test.list_data[0]['scaler'] = dp.preprocess_data(test.list_data[0]['target'])
    data.append({'train': train, 'test': test})
    #if md['make_plots']:
        #hg.plot_train_test(train, test)

### Train network
if md['train_predictor']:
    predictor = fc.train_predictor(data, metadata=md, estimator=md['estimator'])
    for n in range(1):
        if not os.path.isdir(md['serialize_path'] + str(n)):
            os.makedirs(md['serialize_path'] + str(n))
        predictor[n].serialize(Path(md['serialize_path'] + str(n) + "/"))
else:
    ### Load pre-trained predictors
    predictor = fc.load_predictors(md['deserialize_path'], iterations)

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
