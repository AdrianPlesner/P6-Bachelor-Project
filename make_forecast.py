import gluonts
import mxnet as mx
from sklearn import preprocessing
from h5_to_gluonts import load_h5_to_gluon

from gluonts.model.gp_forecaster import GaussianProcessEstimator
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from gluonts.model.predictor import Predictor

data = []
for s in range(4):
    train, test = load_h5_to_gluon("./data/pems-bay.h5", train_size=288 * 3, test_size=288, freq="5Min", sensor=s)
    arr = np.array(train.list_data[0]['target'].reshape(-1, 1))
    scaler = preprocessing.StandardScaler().fit(arr)
    x = scaler.transform(arr)
    arr = np.array(test.list_data[0]['target'].reshape(-1, 1))
    scaler = preprocessing.StandardScaler().fit(arr)
    y = scaler.transform(arr)
    plt.plot(x)
    train.list_data[0]['target'] = x.reshape(-1)
    test.list_data[0]['target'] = y.reshape(-1)
    data.append({'train': train,
                 'test': test})

metadata = {'prediction_length': 288, 'freq': "5Min", 'train_length': 288 * 3}

e = []
for n in range(4):
    e.append(GaussianProcessEstimator(
        metadata['freq'],
        metadata['prediction_length'],
        1,
        Trainer(ctx="cpu",
                epochs=10,
                learning_rate=1e-3,
                hybridize=False,
                num_batches_per_epoch=50),
        metadata['train_length']
    ))

predictor = []
f = []
t = []
f_e = []
t_e = []
for n in range(4):
    ### train data
    p = e[n].train(data[n]['train'])
    ### load pretrained model (model only works for first 4 sensors, 3 days training, 1 day prediction
    #p = Predictor.deserialize(Path("./predictor/p" + str(n)))
    predictor.append(p)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=data[n]['test'],  # test dataset
        predictor=p,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )
    f.append(list(forecast_it))
    t.append(list(ts_it))
    t_e.append(t[n][0])
    f_e.append(f[n][0])


def plot_prob_forecasts(ts_entry, forecast_entry):
    plot_length = metadata['prediction_length'] + metadata['train_length']
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()


for n in range(4):
    plot_prob_forecasts(t_e[n], f_e[n])
