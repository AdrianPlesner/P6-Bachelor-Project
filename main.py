import h5_to_gluonts as hg
import data_processing as dp
import make_forecast as fc
import evaluation

metadata = {'train_length': 288*7, 'test_length': 288, 'freq': "5Min"}
data = []
for n in range(5):
    train, test = hg.load_h5_to_gluon("./data/pems-bay.h5", metadata['train_length'], metadata['test_length'], metadata['freq'], "/speed")
    hg.plot_train_test(train, test)
    data.append({'train': dp.preprocess_data(train), 'test': dp.preprocess_data(test)})
predictor = fc.train_predictor(data, metadata=metadata)
test_data = []
for n in data:
    test_data.append(n['test'])
test, forecast = fc.make_forecast(predictor, test_data, metadata)

for n in range(5):
    fc.plot_prob_forecasts(test[n], forecast[n], n, metadata)

evals = []
for n in range(5):
    e = evaluation.evaluate_forecast(test[n], forecast[n], metadata['prediction_length'])
    evals.append(e)

print(evals)
