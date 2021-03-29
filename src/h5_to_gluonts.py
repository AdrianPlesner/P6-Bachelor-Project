import pandas as pd
from gluonts.core.component import DType
import numpy as np
from gluonts.dataset.common import ListDataset
from gluonts.dataset.util import to_pandas
import matplotlib.pyplot as plt
from gluonts.transform import AddConstFeature


def load_h5_to_gluon(path, freq="1H"):
    store = pd.HDFStore(path)
    train = store["train"],
    valid = store["validation"]
    test = store["test"]
    store.close()

    train = train.swapaxes(0, 1)

    train_data = ListDataset([{
        "start": pd.Timestamp(train.axes[1].array[0], freq=freq),
        "target": row}
        for row in train.values],
        freq=freq
    )
    for n in range(len(train_data.list_data)):
        t = AddConstFeature("sensor_id", "target", 12, train.axes[0].values[n], int)
        train_data.list_data[n] = t.map_transform(train_data.list_data[n], True)

    valid = valid.swapaxes(0, 1)

    valid_data = ListDataset([{
        "start": pd.Timestamp(valid.axes[1].array[0], freq=freq),
        "target": row}
        for row in valid.values],
        freq=freq
    )
    for n in range(len(valid_data.list_data)):
        t = AddConstFeature("sensor_id", "target", 12, valid.axes[0].values[n], int)
        valid_data.list_data[n] = t.map_transform(valid_data.list_data[n], False)

    test = test.swapaxes(0, 1)

    test_data = ListDataset([{
        "start": pd.Timestamp(test.axes[1].array[0], freq=freq),
        "target": row}
        for row in test.values],
        freq=freq
    )
    for n in range(len(test_data.list_data)):
        t = AddConstFeature("sensor_id", "target", 12, test.axes[0].values[n], int)
        test_data.list_data[n] = t.map_transform(test_data.list_data[n], False)
    return train_data, valid_data, test_data


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
