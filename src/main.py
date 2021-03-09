from pathlib import Path
import data_processing as dp
import evaluation as evaluation
import h5_to_gluonts as hg
import make_forecast as fc
import random

metadata = {'train_length': 288*7, 'test_length': 12*4, 'freq': "5Min"}
data = []
iterations = 5
off = 0  # random.randint(0, 7*288 - 12 * 24)
train_p = True

### Load data
for n in range(iterations):
    train, test = hg.load_h5_to_gluon("data/pems-bay.h5", metadata['train_length'], metadata['test_length'], off,
                                      metadata['freq'], "/speed", n)
    # hg.plot_train_test(train, test)
    train.list_data[0]['target'], train.list_data[0]['scaler'] = dp.preprocess_data(train.list_data[0]['target'])
    test.list_data[0]['target'], test.list_data[0]['scaler'] = dp.preprocess_data(test.list_data[0]['target'])
    data.append({'train': train, 'test': test})

### Train network
if train_p:
    predictor = fc.train_predictor(data, metadata=metadata)
    for n in range(iterations):
        predictor[n].serialize(Path("./out-data/p" + str(n) + "/"))
else:
### Load pre-trained predictors
    predictor = fc.load_predictors("./results/5data-7days-1hour-1200its/", iterations)

### Make forecasts
forecast = fc.make_forecast(predictor, data, metadata)
for n in range(iterations):
    data[n], forecast[n] = dp.postprocess_data(data[n], forecast[n])

### Plot forecasts
for n in range(iterations):
    fc.plot_forecast(data[n]['test'], forecast[n], n, metadata)

### Evaluate predictions
evals = []
for n in range(iterations):
    e = evaluation.evaluate_forecast(data[n]['test'], forecast[n], metadata['test_length'])
    evals.append(e)
print(evals)
