from utils.chroma import convert_to_chroma_images
from utils.mel_spectrogram import convert_to_mel_spectrogram_images
from utils.spectrogram import convert_to_spectrogram_images
from utils.mfcc import convert_to_mfcc_images
from preprocessing.split import split_into_3_seconds, split_into_exclusive_datasets
import os


def preprocess(split_use, root_dir=".", split_num = 4):
    """
    Preprocess the data

    Args:
        root_dir (str): path to data directory. Assume that the root_dir has the following structure:
            root_dir
            └── data_dir
                ├── genre1
                │   ├── file1.wav
                │   ├── file2.wav
                │   └── ...
    """
    data_dir = os.path.join(root_dir, "datasources/processed_data")

    if not os.path.isdir(data_dir):
        split_into_3_seconds(os.path.join(root_dir, "datasources"))

    full_dataset= split_into_exclusive_datasets(data_dir)[split_use]
    # TODO: Add function call to convert audio files to features
    convert_to_spectrogram_images(full_dataset, root_dir)
    convert_to_mel_spectrogram_images(full_dataset, root_dir)
    convert_to_chroma_images(full_dataset, root_dir)
    convert_to_mfcc_images(full_dataset, root_dir)