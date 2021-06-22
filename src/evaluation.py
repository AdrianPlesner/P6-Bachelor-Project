import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
import multiprocessing as mp
import scipy.integrate as integrate


def split_validation(data, md):
    """
    Split a data set into slices of prediction length given by metadata
    :param data: ListData object to split
    :param md: Metadata dictionary
    :return: List of ListData objects each containing prediction_length data points
    """
    step = md['prediction_length']
    t = pd.date_range(start=data.list_data[0]['start'], freq=md['freq'], periods=len(data.list_data[0]['target']))
    dum = [ListDataset([{
        'start': t[n],
        'target': d['target'][n:n + step],
        'sensor_id': d['sensor_id'][n:n + step],
        'time_feat': d['time_feat'][::, n:n + step],
        'scaler': d['scaler']
    } for d in data.list_data], freq=md['freq']) for n in range(0, len(t)-11, 1)]
    return dum


def validate_crps(data_slice, forecast):
    """
    Validate forecast against data_slice using CRPS
    :param data_slice: DataSlice to validate against
    :param forecast: Forecast to validate
    :return: Array of CRPS values for each value in forecast
    """
    # Sort samples to estimate CDF
    x = [np.sort(n, 0) for n in forecast.samples]
    evaluation = []
    for n in range(len(x)):
        ar = x[n].swapaxes(0, 1)
        cdf = [CdfShell(a) for a in ar]
        b = crps_vector(data_slice.data[n], cdf)
        evaluation.append(b)
    return np.asarray(evaluation)


validate_crps_vector = np.vectorize(validate_crps, otypes=[list])
"""
Vectorized version of validate_crps()
"""


def validate_mse(data_slice, forecast):
    """
    Validate forecast against data_slice using MSE
    :param data_slice: DataSlice to validate against
    :param forecast: Forecast to validate
    :return: Array of MSE values for each value in forecast
    """
    result = []
    for n in range(len(forecast.mean)):
        val = data_slice.data[n]
        forc = forecast.mean[n]
        result.append(np.array((val-forc)**2))
    return np.asarray(result)


validate_mse_vector = np.vectorize(validate_mse, otypes=[list])
"""
Vectorized version of validate_mse()
"""


def validate_mp(data, forecast, mse=False):
    """
     Validate forecast against data using either CRPS or MSE with multiprocessing to speedup the process
    :param data: List of DataSlice objects to validate against
    :param forecast: List of Forecast objects to validate
    :param mse: whether to use MSE (True) or CRPS (False) for evaluation
    :return: Array of validation values of corresponding to values in forecast
    """
    assert len(data) == len(forecast)
    # Split for number of cpu cores
    n_proc = mp.cpu_count()
    chunk_size = len(data) // n_proc
    rem = len(data) % n_proc
    proc_chunks = []
    # split in chunks for parallelization
    for i_proc in range(n_proc):
        chunkstart = i_proc * chunk_size
        if i_proc < rem:
            chunkstart += i_proc
        else:
            chunkstart += rem
        # make sure to include the division remainder for the last process
        chunkend = (i_proc + 1) * chunk_size
        if i_proc < rem:
            chunkend += i_proc + 1
        else:
            chunkend += rem

        proc_chunks.append((data[chunkstart:chunkend], forecast[chunkstart:chunkend]))

    assert sum([len(x[0]) for x in proc_chunks]) == len(data)

    with mp.Pool(processes=n_proc) as pool:
        # starts the sub-processes without blocking
        # pass the chunk to each worker process
        if mse:
            proc_results = [pool.apply_async(validate_mse_vector, args=(chunk[0], chunk[1],)) for chunk in proc_chunks]
        else:
            proc_results = [pool.apply_async(validate_crps_vector, args=(chunk[0], chunk[1],)) for chunk in proc_chunks]

        result_chunks = []
        # blocks until all results are fetched
        for r in proc_results:
            result_chunks.append(r.get())
    result = np.array([])
    for n in result_chunks:
        result = np.append(result, n)
    return result


def _crps(val, a):
    """
    Computes crps for a value and a distribution
    :param val: the true value
    :param a: a CdfShell object that contains the cumulutative distribution function for a forecast
    :return: the CRPS evaluation for a on val
    """
    x = a.x
    y = a.y
    split = np.searchsorted(x, val)
    if split < 0:
        split = 0
    if split == len(x):
        split -= 1
    lhs = np.square(y[:split])
    rhs = np.square(1 - y[split:])
    if len(lhs) == 0:
        lc = 0
    else:
        lc = integrate.trapezoid(lhs, x[:split])
    if len(rhs) == 0:
        rc = 0
    else:
        rc = integrate.trapezoid(rhs, x[split:])
    return lc + rc


crps_vector = np.vectorize(_crps, otypes=[list])
"""
Vectorised version of _crps()
"""


class CdfShell:
    """
    Object for estimating a Cumulative Distribution Function
    """
    def __init__(self, a):
        """
        Initalize CDF object
        :param a: arrray of samples for which the CDF is estimated
        """
        self.x = a
        self.y = np.arange(len(a)) / float(len(a))

    x = []
    y = []

    def cdf(self, a):
        """
        CDF function of the object
        :param a: CDF input
        :return: Corresponding CDF output
        """
        v = np.searchsorted(self.x, a, 'left')
        if v == len(self.y):
            return 1.0
        return self.y[v]


class Forecast:
    """
    A Forecast object containing samples and mean value of a forecast
    """
    def __init__(self, f, m):
        """
        :param f: Expects a 3d array/list with dimensions (n,m,o)
        n is the number of sensors i.e. 325 sensors
        m is the number of samples per sensor i.e. 250 samples
        o is the prediction length i.e. 12 data point
        mean is a n x o array
        :param m: array of mean values should have dimensions (n,o)
        """
        self.samples = f
        self.mean = m

    def extend(self, f):
        """
        Append forecast objects together
        :param f: Forecast object to append to this one
        :return:
        """
        assert len(self.samples), len(f.samples)
        for n in range(len(self.samples)):
            self.samples[n] = np.append(self.samples[n], f.samples[n], axis=1)
            self.mean[n] = np.append(self.mean[n], f.mean[n])


class DataSlice:
    """
    Contains a slice of data
    """
    def __init__(self, data):
        """
        :param data: Expects a 2d array/list with dimensions (n,m)
        n is the number of sensors i.e. 325 sensors
        m is the data length i.e. 12 data points
        """
        self.data = data
