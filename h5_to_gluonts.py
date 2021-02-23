import pandas as pd

from gluonts.dataset.common import ListDataset
from gluonts.dataset.util import to_pandas
import matplotlib.pyplot as plt


def load_h5_to_gluon(path, train_size=0, test_size=0, freq="1H", key=""):
    store = pd.HDFStore(path)
    if key == "":
        key = store.keys()[0]
    data = store[key]
    store.close()
    sensor = 0
    size = len(data.values)
    if train_size == 0 and test_size == 0:
        train_size = size - size // 10
        test_size = size - train_size
    elif train_size == 0:
        train_size = size - test_size
    elif test_size == 0:
        test_size = size - train_size
    data_points = train_size
    test_points = test_size + train_size

    data_start = [pd.Timestamp(data.axes[0].array[0], freq=freq) for _ in range(test_points)]

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
    return train_data, test_data


def plot_train_test(train_data, test_data, size_x=20, size_y=10):
    train_entry = next(iter(train_data))

    test_entry = next(iter(test_data))

    train_series = to_pandas(train_entry)
    test_series = to_pandas(test_entry)

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(size_x, size_y))

    train_series.plot(ax=ax[0])
    ax[0].grid(which="both")
    ax[0].legend(["train series"], loc="upper left")

    test_series.plot(ax=ax[1])
    ax[1].axvline(train_series.index[-1], color='r')  # end of train dataset
    ax[1].grid(which="both")
    ax[1].legend(["test series", "end of train series"], loc="upper left")

    plt.show()
    return
