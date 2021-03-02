import h5_to_gluonts as hg
import data_processing as dp
import make_forecast as fc
import evaluation

metadata = {'train_length': 288*3, 'test_length': 288, 'freq': "5Min"}
data = []
for n in range(4):
    train, test = hg.load_h5_to_gluon("./data/pems-bay.h5", metadata['train_length'], metadata['test_length'],
                                      metadata['freq'], "/speed", n)
    #hg.plot_train_test(train, test)
    train.list_data[0]['target'] = dp.preprocess_data(train.list_data[0]['target'])
    test.list_data[0]['target'] = dp.preprocess_data(test.list_data[0]['target'])
    data.append({'train': train , 'test': test})
#predictor = fc.train_predictor(data, metadata=metadata)
predictor = fc.load_predictors("./predictor/", 4)
test_data = []
for n in data:
    test_data.append(n['test'])
test, forecast = fc.make_forecast(predictor, test_data, metadata)

for n in range(4):
    fc.plot_prob_forecasts(test[n], forecast[n], n, metadata)

evals = []
for n in range(4):
    e = evaluation.evaluate_forecast(test[n], forecast[n], metadata['test_length'])
    evals.append(e)

print(evals)
