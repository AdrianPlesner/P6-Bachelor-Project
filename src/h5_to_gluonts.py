import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.transform import AddConstFeature
from gluonts.transform import AddTimeFeatures
import gluonts.time_feature as time
import numpy as np


def load_h5_to_gluon(md):
    """
    Load data from a .h5 archive to GluonTS ListData. The archive should have the keys 'train', 'validation', 'test',
    'locations'
    :param md: metadata dictionary containing at least keys 'path' and 'freq'
    :return: a tuple containing train, validation and test ListData objects
    """
    store = pd.HDFStore(md['path'])
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
            idx = np.where(loc.axes[0].values == str(data.axes[0].values[n]))[0][0]
            t = AddConstFeature("lat", "target", 12, loc.values[idx, 0])
            list_data.list_data[n] = t.map_transform(list_data.list_data[n], b)
            t = AddConstFeature("long", "target", 12, loc.values[idx, 1])
            list_data.list_data[n] = t.map_transform(list_data.list_data[n], b)
        t = AddTimeFeatures("start", "target", "time_feat", [time.DayOfWeek(), time.HourOfDay(), time.MinuteOfHour()],
                            md['prediction_length'])
        for n in range(len(list_data.list_data)):
            list_data.list_data[n] = t.map_transform(list_data.list_data[n], b)
        out.append(list_data)

    return out[0], out[1], out[2]
