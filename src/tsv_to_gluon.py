import csv
import os
import sys

tsv_file = open("data/Earthquakes_TEST.tsv")
data = csv.reader(tsv_file, delimiter="\t")

