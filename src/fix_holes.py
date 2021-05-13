import pandas as pd
import numpy as np


class DateList:
    def __init__(self, dow, hod, moh, sensor):
        self.day_of_week = dow
        self.hour_of_day = hod
        self.minute_of_hour = moh
        self.sensor = sensor
        self.values = []
        self.mean = 0
        self.std = 0

    def add_val(self, n: float):
        self.values.append(n)
        self.mean = np.mean(self.values)
        self.std = np.std(self.values)

    def draw_val(self, num):
        return np.random.normal(self.mean, self.std, num)


store = pd.HDFStore("data/metr-la.h5")

data = store['df']
data = data.T

sensors = []
for s in range(len(data.values)):
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

for s in range(len(data.values)):
    i = 3
    j = 0
    k = 0
    sens = data.values[s]
    for n in sens:
        d = k + 12 * j + 12 * 24 * i
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

for n in range(len(data.values)):
    d = data.values[n]
    for m in range(len(d)):
        e = d[m]
        if e <= 0:
            date = data.axes[1][m]
            day_list = sensors[n][date.day_of_week][date.hour][date.minute // 5]
            data.values[n, m] = day_list.draw_val(1)

data = data.T

train = data[:23904]
validation = data[23904:27360]
test = data[27360:]
store.put('train', train)
store.put('validation', validation)
store.put('test', test)
store.close()

