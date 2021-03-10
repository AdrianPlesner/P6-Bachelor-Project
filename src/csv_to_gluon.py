import csv
import pandas as pd
from gluonts.dataset.common import ListDataset
import numpy as np


def load_csv_to_gluon(path, train_length=0, test_length=0, freq="1D", offset=0, location=0, total=False):
    offset += 2
    test_length += train_length
    file = open(path, newline='')
    reader = csv.reader(file)
    a = [row for row in reader]
    file.close()
    if total:
        train_data = np.sum(a[1:][offset:offset+train_length], axis=1)
        test_data = np.sum(a[1:][offset:offset+test_length], axis=1)
    else:
        train_data = a[location][offset:offset+train_length]
        test_data = a[location][offset:offset+test_length]
    return ListDataset([{'start': pd.Timestamp(a[0][offset]), 'target': train_data}], freq), \
        ListDataset([{'start': pd.Timestamp(a[0][offset]), 'target': test_data}], freq)
