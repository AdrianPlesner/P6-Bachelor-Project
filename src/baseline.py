import evaluation as ev
import h5_to_gluonts as hg
from data_processing import listdata_to_array
import numpy as np
from evaluation import Forecast


class DateList:
    """
    A class for keeping all values with the same day of the week, hour of the day, minute of the hour, and sensor in
     order to compute historical average baseline
    """
    def __init__(self, dow, hod, moh, sensor):
        """
        Create a DeteList instance
        :param dow: day of the week [0-6]
        :param hod: hour of the day [0-23]
        :param moh: minute of the hour [0-59]
        :param sensor: sensor
        """
        self.day_of_week = dow
        self.hour_of_day = hod
        self.minute_of_hour = moh
        self.sensor = sensor
        self.values = []
        self.mean = 0
        self.std = 0

    def add_val(self, n: float):
        """
        Adds a value to the list of values and recompute the mean and std
        :param n: value to be added
        :return:
        """
        self.values.append(n)
        self.mean = np.mean(self.values)
        self.std = np.std(self.values)

    def draw_val(self, num):
        """
        Draws a number of random values from the gaussian distribution made of the mean an std of values
        :param num: the amount of random values to draw
        :return: numpy array containing the values drawn
        """
        return np.random.normal(self.mean, self.std, num)


def make_forecast(slice, sensors):
    ts = slice.list_data[0]['start']
    result = []
    for n in range(len(slice.list_data)):
        f = []
        for m in range(12):
            f.append(sensors[n][ts.day_of_week][ts.hour][m].draw_val(250))
        result.append(np.stack(f).transpose())
    samples = np.stack(result)
    mean = [[np.mean(sensor[::, m]) for m in range(12)] for sensor in samples]
    return Forecast(samples, mean)


def compute_baseline(path):
    """
    Compute histroical average baseline gor at given data set
    :param path: path to data file
    :return: a tuple containing arrays of CRPS and MSE evaluation of the HA baseline
    """
    md = {'freq': "5Min", 'prediction_length': 12, 'path': path}
    train, valid, test = hg.load_h5_to_gluon(md['path'], md)
    for n in test.list_data:
        n['scaler'] = None
    # Create DateList objects
    sensors = []
    for s in range(len(train.list_data)):
        days = []
        for i in range(7):
            hours = []
            for j in range(24):
                minutes = []
                for k in range(0, 60, 5):
                    minutes.append(DateList(i, j, k, s))
                hours.append(minutes)
            days.append(hours)
        sensors.append(days)
    # put values from training data into DateList objects
    for s in range(len(train.list_data)):
        ### Beware i,j,k are harcoded values for the weekday, hour and minute of the first data point
        i = 6
        j = 0
        k = 0
        sens = train.list_data[s]
        for n in sens['target']:
            if n > 0:
                sensors[s][i][j][k].add_val(n)
            k += 1
            if k >= 12:
                j += 1
                k = 0
                if j >= 24:
                    i += 1
                    j = 0
                    if i >= 7:
                        i = 0
    test_slices = ev.split_validation(test, md)
    data_slices = listdata_to_array(test_slices)
    ss = []
    # Split data for less memory consumption
    while len(test_slices) > 100:
        ss.append((test_slices[:100], data_slices[:100]))
        test_slices = test_slices[100:]
        data_slices = data_slices[100:]
    ss.append((test_slices, data_slices))
    test_slices = None
    data_slices = None
    # Validate forecasts
    mse = []
    crps = []
    for slices in ss:
        forecasts = [make_forecast(slice, sensors) for slice in slices[0]]
        mse.append(np.average(np.stack(ev.validate_mp(slices[1], forecasts, mse=True))[::, ::, 11]))
        crps.append(np.average(np.stack(ev.validate_mp(slices[1], forecasts, mse=False))[::, ::, 11]))
    print(f'crps: {np.average(np.stack(crps))}, mse: {np.average(np.stack(mse))}')
    return crps, mse
