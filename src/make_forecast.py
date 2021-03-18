from gluonts.model.gp_forecaster import GaussianProcessEstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
import matplotlib.pyplot as plt
from pathlib import Path
from gluonts.model.predictor import Predictor
import data_processing as dp
import numpy as np
import pandas as pd
import mxnet as mx


def train_predictor(data=None, test_length=0, freq="1H", train_length=0, metadata=None, estimator=None):
    if metadata is None:
        metadata = {}
    if data is None:
        data = [{}]
    trainer = Trainer(ctx=mx.context.gpu(),
                      epochs=8,
                      learning_rate=1e-3,
                      hybridize=False,
                      num_batches_per_epoch=300)
    if estimator is None or estimator == "GP":
        estimator = GaussianProcessEstimator(
            metadata['freq'],
            metadata['test_length'],
            5,
            trainer,
            metadata['train_length']
        )
    elif estimator == "rnn":
        estimator = DeepFactorEstimator(
            metadata['freq'],
            metadata['test_length'],
            trainer=trainer,
            context_length=metadata['train_length'],
            cardinality=list([5])
        )
    elif estimator == "AR":
        estimator = DeepAREstimator(
            metadata['freq'],
            metadata['test_length'],
            trainer,
            metadata['train_length']
        )
    else:
        print("Estimator error")
        exit()

    assert (len(data) > 0)

    if "test_length" not in metadata.keys():
        metadata['test_length'] = test_length
    assert (metadata['test_length'] > 0)

    if "train_length" not in metadata.keys():
        metadata['train_length'] = train_length
    assert (metadata['train_length'] > 0)
    if "freq" not in metadata.keys():
        metadata['freq'] = freq
    metadata['data_sets'] = len(data)
    e = []
    predictor = []
    d = data[0]['train']
    for n in range(1, metadata['data_sets']):
        d.list_data.extend(data[n]['train'].list_data)
    p = estimator.train(d)
    predictor.append(p)
    # for n in range(metadata['data_sets']):
    #     e.append(estimator)
    #     p = e[n].train(data[n]['train'])
    #     predictor.append(p)

    return predictor


def make_forecast(predictor, data, metadata):
    metadata['data_sets'] = len(data)
    f = []
    for n in range(metadata['data_sets']):
        f.append(list(predictor[0].predict(data[n]['train']))[0])
    return f


# def make_forecast(predictor, data, metadata):
#     metadata['data_sets'] = len(data)
#     f = []
#     t = []
#     f_e = []
#     t_e = []
#     for n in range(metadata['data_sets']):
#         forecast_it, ts_it = make_evaluation_predictions(
#             dataset=data[n],  # test dataset
#             predictor=predictor[n],  # predictor
#             num_samples=100,  # number of sample paths we want for evaluation
#         )
#         f.append(list(forecast_it))
#         t.append(list(ts_it))
#         t_e.append(t[n][0])
#         f_e.append(f[n][0])
#     return t_e, f_e


# def plot_prob_forecasts(ts_entry, forecast_entry, num, metadata):
#     plot_length = metadata['test_length'] + metadata['train_length']
#     prediction_intervals = (90.0, 50.0)
#     legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
#
#     fig, ax = plt.subplots(1, 1, figsize=(20, 10))
#     ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
#     forecast_entry.plot(prediction_intervals=(), color='#008000')
#     y1, y2 = dp.make_prediction_interval(forecast_entry.mean, 0.67)
#     plt.fill_between(ts_entry[metadata['train_length']:plot_length], y1, y2, color='#00800080')
#     y1, y2 = dp.make_prediction_interval(forecast_entry.mean, 1.64)
#     plt.fill_between(ts_entry[metadata['train_length']:plot_length], y1, y2, color='#00800060')
#     plt.grid(which="both")
#     plt.legend(legend, loc="upper left")
#     plt.title("dataset " + str(num))
#     plt.show()
#     #plt.savefig("out-data/plot" + str(num))


def plot_forecast(lst_data, forecast_entry, num, metadata, offset=0, path="", sensor=-1):
    if sensor == -1:
        sensor = num
    plot_length = metadata['test_length'] + metadata['train_length']
    prediction_intervals = (90.0, 50.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
    data = [pd.date_range(start=lst_data.list_data[0]['start'], freq=metadata['freq'], periods=plot_length),
            lst_data.list_data[0]['target']]
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    plt.plot(data[0][offset:plot_length], data[1][offset:plot_length])
    plt.plot(forecast_entry.index, forecast_entry.median, color='#008000')
    y1, y2 = dp.make_prediction_interval(forecast_entry, 0.67)
    plt.fill_between(data[0][metadata['train_length']:plot_length], y1, y2, color='#00800080')
    y1, y2 = dp.make_prediction_interval(forecast_entry, 1.64)
    plt.fill_between(data[0][metadata['train_length']:plot_length], y1, y2, color='#00800060')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.title("dataset " + str(sensor))
    plt.savefig(metadata['deserialize_path'] + "pictures/" + str(num) + "/" + path)
    plt.close()


def load_predictors(path, num, sub_paths=None):
    if sub_paths is None:
        sub_paths = []
        for n in range(num):
            sub_paths.append("p" + str(n))

    predictor = []
    for n in range(num):
        p = Predictor.deserialize(Path(path + sub_paths[n]))
        p.prediction_net.ctx = p.ctx
        predictor.append(p)

    return predictor
