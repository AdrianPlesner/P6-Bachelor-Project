import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.util import to_pandas
import matplotlib.pyplot as plt

store = pd.HDFStore('./pems-bay.h5')
data = store['/speed']
store.close()
data_points = 288 * 31
test_points = data_points + 288
sensor = 0
freq = '5Min'
data_start = [pd.Timestamp("01-01-2019", freq=freq) for _ in range(test_points)]

train_data = ListDataset([{
    "start": start,
    "target": target}
    for (start, target) in zip(data_start[data_points:], [data.values[:data_points, sensor]])],
    freq=freq
)
test_data = ListDataset([{
    "start": start,
    "target": target}
    for (start, target) in zip(data_start[:test_points], [data.values[:test_points, sensor]])],
    freq=freq
)
train_entry = next(iter(train_data))

test_entry = next(iter(test_data))

train_series = to_pandas(train_entry)
test_series = to_pandas(test_entry)

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(20, 10))

train_series.plot(ax=ax[0])
ax[0].grid(which="both")
ax[0].legend(["train series"], loc="upper left")

test_series.plot(ax=ax[1])
ax[1].axvline(train_series.index[-1], color='r')  # end of train dataset
ax[1].grid(which="both")
ax[1].legend(["test series", "end of train series"], loc="upper left")

plt.show()
