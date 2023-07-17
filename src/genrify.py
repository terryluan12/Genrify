from datasources import download_datasets
from preprocessing import preprocess
import os

# Change to subset used to train the CNN
subset_num = None
method = None


if not os.path.isdir("datasources/Data") and not os.path.isdir("datasources/processed_data"):
    download_datasets()

preprocess(subset_num, method)