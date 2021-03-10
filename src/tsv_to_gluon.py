import csv
import pandas as pd
from gluonts.dataset.common import ListDataset
import os
import sys


def load_tsv_to_gluon(path, train_length=0, test_length=0, sensor=0, freq='1H', start_date=0):
    if start_date == 0:
        start_date = pd.Timestamp.today()
    test_length += train_length
    file = open(path, newline='')
    reader = csv.reader(file, delimiter="\t")
    data = [row[1:] for row in reader]
    file.close()
    return ListDataset([{'start': start_date, 'target': data[sensor][:train_length]}], freq), \
           ListDataset([{'start': start_date, 'target': data[sensor][:test_length]}], freq)

