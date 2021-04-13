from gluonts.model.gp_forecaster import GaussianProcessEstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.distribution import StudentTOutput, MultivariateGaussianOutput, LowrankMultivariateGaussianOutput
from gluonts.mx.kernels import RBFKernelOutput, PeriodicKernelOutput
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
import matplotlib.pyplot as plt
from pathlib import Path
from gluonts.model.predictor import Predictor
import data_processing as dp
import numpy as np
import pandas as pd
import mxnet as mx


def train_predictor(data=None, md=None):
    if md is None:
        exit("Missing metadata for training")
    if data is None:
        exit("Missing data for training")
    trainer = Trainer(ctx=mx.context.gpu(),
                      epochs=250,
                      batch_size=32,
                      learning_rate=1e-3,
                      hybridize=False,
                      num_batches_per_epoch=1143)
    if md['estimator'] is None or md['estimator'] == "GP":
        if md['kernel'] == "RBF":
            kernel = RBFKernelOutput()
        else:
            kernel = PeriodicKernelOutput()
        estimator = GaussianProcessEstimator(
            freq=md['freq'],
            prediction_length=md['prediction_length'],
            context_length=md['prediction_length'],
            cardinality=md['sensors'],
            trainer=trainer,
            kernel_output=kernel

        )
    elif md['estimator'] == "DeepFactor":
        if md['distribution'] == "StudentT":
            distribution = StudentTOutput()
        elif md['distribution'] == "Gaussian":
            distribution = MultivariateGaussianOutput()
        elif md['distribution'] == "Low-rank gaussian":
            distribution = LowrankMultivariateGaussianOutput()
        else:
            distribution = None
            exit("Missing distribution")

        estimator = DeepFactorEstimator(
            freq=md['freq'],
            prediction_length=md['prediction_length'],
            trainer=trainer,
            context_length=md['prediction_length'],
            cardinality=list(md["sensors"]),
            num_hidden_global=md['global_units'],
            num_layers_global=md['global_layers'],
            num_factors=md['factors'],
            num_hidden_local=md['local_units'],
            num_layers_local=md['local_layers'],
            cell_type=md["cell_type"],
            distr_output=distribution

        )
    elif md['estimator'] == "DeepAR":
        estimator = DeepAREstimator(
            freq=md['freq'],
            prediction_length=md['prediction_length'],
            trainer=trainer,
            context_length=md['prediction_length'],
            num_layers=md['layers'],
            num_cells=md['cells'],
            cell_type=md['cell_type'],
            dropout_rate=md['dropout_rate']
        )
    else:
        estimator = None
        exit("Invalid estimator")

    assert (len(data) > 0)

    return estimator.train(data)


def make_forecast(predictor, data):
    return list(predictor.predict(data, num_samples=100))


make_forecast_vector = np.vectorize(make_forecast, otypes=[list])


# def make_forecast(predictor, data, md):
#     md['data_sets'] = len(data)
#     f = []
#     t = []
#     f_e = []
#     t_e = []
#     for n in range(md['data_sets']):
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


# def plot_prob_forecasts(ts_entry, forecast_entry, num, md):
#     plot_length = md['test_length'] + md['train_length']
#     prediction_intervals = (90.0, 50.0)
#     legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
#
#     fig, ax = plt.subplots(1, 1, figsize=(20, 10))
#     ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
#     forecast_entry.plot(prediction_intervals=(), color='#008000')
#     y1, y2 = dp.make_prediction_interval(forecast_entry.mean, 0.67)
#     plt.fill_between(ts_entry[md['train_length']:plot_length], y1, y2, color='#00800080')
#     y1, y2 = dp.make_prediction_interval(forecast_entry.mean, 1.64)
#     plt.fill_between(ts_entry[md['train_length']:plot_length], y1, y2, color='#00800060')
#     plt.grid(which="both")
#     plt.legend(legend, loc="upper left")
#     plt.title("dataset " + str(num))
#     plt.show()
#     #plt.savefig("out-data/plot" + str(num))


def plot_forecast(lst_data, forecast_entry, num, md, offset=0, path="", sensor=-1):
    if sensor == -1:
        sensor = num
    plot_length = md['test_length'] + md['train_length']
    prediction_intervals = (90.0, 50.0)
    legend = ["observations", "mean prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
    data = [pd.date_range(start=lst_data.list_data[0]['start'], freq=md['freq'], periods=plot_length),
            lst_data.list_data[0]['target']]
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    plt.plot(data[0][offset:plot_length], data[1][offset:plot_length])
    plt.plot(forecast_entry.index, forecast_entry.mean, color='#008000')
    y1, y2 = dp.make_prediction_interval(forecast_entry, 0.67)
    plt.fill_between(data[0][md['train_length']:plot_length], y1, y2, color='#00800080')
    y1, y2 = dp.make_prediction_interval(forecast_entry, 1.64)
    plt.fill_between(data[0][md['train_length']:plot_length], y1, y2, color='#00800060')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.title("dataset " + str(sensor))
    plt.show()
    #    plt.savefig(md['deserialize_path'] + "pictures/" + str(num) + "/" + path)
    plt.close()


def load_predictor(path):
    p = Predictor.deserialize(Path(path))
    p.prediction_net.ctx = p.ctx

    return p
