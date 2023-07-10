from utils.chroma import convert_to_chroma_images
from utils.mel_spectrogram import convert_to_mel_spectrogram_images
from utils.spectrogram import convert_to_spectrogram_images
from utils.mfcc import convert_to_mfcc_images


def preprocess(data_dir):
    """
    Preprocess the data

    Args:
        data_dir (str): path to data directory. Assume that the data_dir has the following structure:
            data_dir
            ├── genre1
            │   ├── file1.wav
            │   ├── file2.wav
            │   └── ...
    """

    # TODO: Add function call to split audio files into 3 second clips
    # TODO: Add function call to convert audio files to features
    convert_to_spectrogram_images(data_dir)
    convert_to_mel_spectrogram_images(data_dir)
    convert_to_chroma_images(data_dir)
    convert_to_mfcc_images(data_dir)