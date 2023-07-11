from datasources import download_datasets
from preprocessing import preprocess
import os


if not os.path.isdir("datasources/processed_data"):
    download_datasets()

preprocess("datasources")