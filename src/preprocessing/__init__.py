from utils.chroma import convert_to_chroma_images
from utils.mel_spectrogram import convert_to_mel_spectrogram_images
from utils.spectrogram import convert_to_spectrogram_images
from utils.mfcc import convert_to_mfcc_images
from preprocessing.split import split_into_3_seconds
import os


def preprocess(root_dir="."):
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

    split_into_3_seconds(os.path.join(root_dir, "datasources"))
    # TODO: Add function call to convert audio files to features
    convert_to_spectrogram_images(data_dir)
    convert_to_mel_spectrogram_images(data_dir)
    convert_to_chroma_images(data_dir)
    convert_to_mfcc_images(data_dir)