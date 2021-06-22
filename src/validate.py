import data_processing as dp
import evaluation as evaluation
import h5_to_gluonts as hg
import make_forecast as fc
import numpy as np
import sys
import json
import csv
from evaluation import Forecast

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
    print(str(md))
    print("loading data...")
    train, valid, test = hg.load_h5_to_gluon(md)
    train = None
    valid = None
    if md['normalize']:
        for data in (test,):
            for n in range(len(data)):
                data.list_data[n]['target'], data.list_data[n]['scaler'] = dp.preprocess_data(
                    data.list_data[n]['target'])
    predictor = fc.load_predictor(md['serialize_path'], md)
    test_slices = evaluation.split_validation(test, md)
    test = None
    ss = []
    while len(test_slices) > 100:
        ss.append(test_slices[:100])
        test_slices = test_slices[100:]
    if len(test_slices) > 1:
        ss.append(test_slices)
    test_slices = None
    crps = []
    mse = []
    i = 1
    for slices in ss:
        print(f'{i} of {len(ss)}')
        i += 1
        print("making predictions...")
        forecast = fc.make_forecast_vector(predictor, slices, md)
        if md['estimator'] == "TempFlow":
            forecast = [
                Forecast([slice[0].samples[::, ::, n] for n in range(md['sensors'])], [slice[0].mean[::, n] for n in range(md['sensors'])])
                for slice in forecast]
        else:
            forecast = [Forecast([sensor.samples for sensor in slice], [sensor.mean for sensor in slice]) for slice in
                        forecast]
        print("Rescaling...")
        slices, forecast = dp.postprocess_data_vector(slices, forecast)
        slices = dp.listdata_to_array(slices)
        print("evaluating...")
        evals = np.stack(evaluation.validate_mp(slices[1:], forecast[:len(forecast) - 1], mse=False))
        crps.append(evals)
        evals = np.stack(evaluation.validate_mp(slices[1:], forecast[:len(forecast) - 1], mse=True))
        mse.append(evals)
    cjoin = crps[0]
    for i in crps[1:]:
        cjoin = np.append(cjoin, i, 0)
    crps = np.average(cjoin, 0)
    mjoin = mse[0]
    for i in mse[1:]:
        mjoin = np.append(mjoin, i, 0)
    mse = np.average(mjoin, 0)

    f = open(md['serialize_path'] + "crps.csv", 'w', newline='')
    writer = csv.writer(f)
    writer.writerows(crps)
    f.close()
    f = open(md['serialize_path'] + "mse.csv", 'w', newline='')
    writer = csv.writer(f)
    writer.writerows(mse)
    f.close()
