import csv
import pandas as pd
from gluonts.dataset.common import ListDataset
import numpy as np


def load_csv_to_gluon(path, train_length=0, test_length=0, freq="1D", offset=0, location=0, total=False):
    offset += 4
    test_length += train_length
    file = open(path, newline='')
    reader = csv.reader(file)
    a = [row for row in reader]
    b = a[1:]
    b = [d[offset:] for d in b]
    b = np.array(b, dtype=int)
    file.close()
    if total:
        train_data = np.sum(b[1:][:train_length], axis=0)
        test_data = np.sum(b[1:][:test_length], axis=0)
    else:
        train_data = b[location][:train_length]
        test_data = b[location][:test_length]
    return ListDataset([{'start': pd.Timestamp(a[0][offset]), 'target': train_data}], freq), \
        ListDataset([{'start': pd.Timestamp(a[0][offset]), 'target': test_data}], freq)
