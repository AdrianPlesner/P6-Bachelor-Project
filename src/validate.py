import data_processing as dp
import evaluation as evaluation
import h5_to_gluonts as hg
import make_forecast as fc
import numpy as np
import sys
import json
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
        crps.append(np.average(evals[::, ::, 11]))
        evals = np.stack(evaluation.validate_mp(slices[1:], forecast[:len(forecast) - 1], mse=True))
        mse.append((np.average(evals[::, ::, 11])))
        print(crps)
        print(mse)
    crps = np.array(crps)
    crps = np.average(crps)
    mse = np.array(mse)
    mse = np.average(mse)
    print(f"evaluation on test is CRPS: {crps}, MSE: {mse}")
