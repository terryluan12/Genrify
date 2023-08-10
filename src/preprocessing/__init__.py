from utils.chroma import convert_to_chroma_images
from utils.mel_spectrogram import convert_to_mel_spectrogram_images
from utils.spectrogram import convert_to_spectrogram_images
from utils.mfcc import convert_to_mfcc_images
from preprocessing.split import split_into_3_seconds, split_test_data_into_3_seconds, split_into_exclusive_datasets
from torchvision.datasets import DatasetFolder
import os
import librosa


def preprocess(split_use, method, root_dir=".", split_num = 4, is_test=False):
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

    if method!="create_testing_data":
        data_dir = os.path.join(root_dir, "datasources/processed_data")
        if not os.path.isdir(data_dir):
            split_into_3_seconds(os.path.join(root_dir, "datasources"))
        full_dataset= split_into_exclusive_datasets(data_dir)[split_use]
        # TODO: Add function call to convert audio files to features
        if method == "spec":
            convert_to_spectrogram_images(full_dataset, root_dir, training=True and not is_test)
        elif method == "mel":
            convert_to_mel_spectrogram_images(full_dataset, root_dir, training=True and not is_test)
        elif method == "chroma":
            convert_to_chroma_images(full_dataset, root_dir, training=True and not is_test)
        elif method == "mfcc":
            convert_to_mfcc_images(full_dataset, root_dir, training=True and not is_test)
        else:
            raise Exception("Must be spec, mel, chroma, or mfcc")
    else:
        data_dir = os.path.join(root_dir, "datasources/processed_test_data")
        split_test_data_into_3_seconds("/content/Genrify/src/datasources")
        full_dataset=DatasetFolder(data_dir, librosa.load, extensions=[".wav"])
        convert_to_spectrogram_images(full_dataset, root_dir, training=False)
        convert_to_mel_spectrogram_images(full_dataset, root_dir, training=False)
        convert_to_chroma_images(full_dataset, root_dir, training=False)
        convert_to_mfcc_images(full_dataset, root_dir, training=False)
    