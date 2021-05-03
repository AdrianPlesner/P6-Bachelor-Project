import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.util import to_pandas
import matplotlib.pyplot as plt
from gluonts.transform import AddConstFeature
from gluonts.transform import AddTimeFeatures
from gluonts.transform import ListFeatures
import gluonts.time_feature as time


def load_h5_to_gluon(path, md):
    store = pd.HDFStore(path)
    train = store["train"]
    valid = store["validation"]
    test = store["test"]
    loc = store["locations"]
    store.close()
    out = []
    for (data, b) in [(train, True), (valid, False), (test, False)]:
        data = data.swapaxes(0, 1)
        list_data = ListDataset([{
            "start": pd.Timestamp(data.axes[1].array[0], freq=md['freq']),
            "target": row}
            for row in data.values],
            freq=md['freq']
        )
        for n in range(len(list_data.list_data)):
            t = AddConstFeature("sensor_id", "target", 12, data.axes[0].values[n], int)
            list_data.list_data[n] = t.map_transform(list_data.list_data[n], b)
            idx = loc.axes[0].index(data.axes[0].values[n])
            t = AddConstFeature("lat", "target", 12, loc.values[idx, 0], int)
            list_data.list_data[n] = t.map_transform(list_data.list_data[n], b)
            t = AddConstFeature("long", "target", 12, loc.values[idx, 1], int)
            list_data.list_data[n] = t.map_transform(list_data.list_data[n], b)
            t = ListFeatures("feat_static_real", ["sensor_id", "lat", "long"], False)
            list_data.list_data[n] = t.map_transform(list_data.list_data[n], b)
        t = AddTimeFeatures("start", "target", "time_feat", [time.DayOfWeek(), time.HourOfDay(), time.MinuteOfHour()], md['prediction_length'])
        for n in range(len(list_data.list_data)):
            list_data.list_data[n] = t.map_transform(list_data.list_data[n], b)
        out.append(list_data)

    return out[0], out[1], out[2]


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
