from pathlib import Path

import data_processing as dp
import evaluation
import h5_to_gluonts as hg
import make_forecast as fc

metadata = {'train_length': 288*3, 'test_length': 288, 'freq': "5Min"}
data = []
iterations = 1
for n in range(iterations):
    train, test = hg.load_h5_to_gluon("./data/pems-bay.h5", metadata['train_length'], metadata['test_length'],
                                      metadata['freq'], "/speed", n)
    #hg.plot_train_test(train, test)
    train.list_data[0]['target'], train.list_data[0]['scaler'] = dp.preprocess_data(train.list_data[0]['target'])
    test.list_data[0]['target'], test.list_data[0]['scaler'] = dp.preprocess_data(test.list_data[0]['target'])
    data.append({'train': train, 'test': test})

# predictor = fc.train_predictor(data, metadata=metadata)
# for n in range(iterations):
#     predictor[n].serialize(Path("./out-data/p" + str(n) + "/"))
predictor = fc.load_predictors("./predictor/out-data/", iterations)


forecast = fc.make_forecast(predictor, data, metadata)
for n in range(iterations):
    data[n], forecast[n] = dp.postprocess_data(data[n], forecast[n])


fc.plot_forecast(data[0]['test'], forecast[0], 0, metadata)
# evals = []
# for n in range(iterations):
#     e = evaluation.evaluate_forecast(test[n], forecast[n], metadata['test_length'])
#     evals.append(e)
#
# print(evals)
