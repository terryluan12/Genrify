import librosa
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
from augment_sample import augment_sample
matplotlib.use('Agg')

def extract_features_mel_spectrogram(datapoint, training=False):
    """
    Extract mel-spectrogram features from an audio file.

    Args:
        datapoint (tuple(numpy.ndarray, int)): A tuple containing the audio data as a numpy array and the sample rate as an integer.
        training (bool): Whether the extraction is for training data or not.

    Returns:
        (np.array or list of np.array): Mel-spectrogram features in dB. If training is True, returns a list of spectrogram features where each item is of shape (number of frequency bins) x (number of time frames). If training is False, returns a single spectrogram feature of shape (number of frequency bins) x (number of time frames).
    """
    audio, sample_rate = datapoint
    if training:
        augmented_mel_list = []
        audios = augment_sample(audio, sample_rate)
        for audio in audios:
            mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            augmented_mel_list.append(mel_spectrogram_db)
        return augmented_mel_list
    else:
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        return mel_spectrogram_db


def save_mel_spectrogram_image(mel_spectrogram_data, file_name):
    """
    Save mel-spectrogram data as an image

    Args:
        spectrogram_data (np.array): Spectrogram features in dB (number of frequncy bins) x (number of time frames) 
        file_name (str): path to save image
    """
    if mel_spectrogram_data is None:
        return
    plt.figure(figsize=(5, 4), frameon=False)
    plt.ylim(0, 11000) # Always plot up to 11kHz
    librosa.display.specshow(mel_spectrogram_data, x_axis='time', y_axis='hz', vmin=-40, vmax=40) # Always show magnitude from -40 to +40 dB
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.cla()
    plt.close('all')


def convert_to_mel_spectrogram_images(datasets, root_dir=".", training=False):
    """
    Converts the WAV files to Mel Spectrogram features and saves them as images in a new directory structure.

    Args:
        datasets (List): List of datasets that are being converted to Mel Spectrogram features.
        root_dir (string): Path to the source directory of the Genrify module.
        training (bool): Whether the conversion is for training or not.
    """
    print(f'Converting to Mel-Spectrogram')
    processed_data_dir = os.path.join(root_dir, "datasources/mel")
    os.makedirs(processed_data_dir, exist_ok=True)
    
    i = 0
    for dataset in datasets:
        for data, label in dataset:
            os.makedirs(os.path.join(processed_data_dir, str(label)), exist_ok=True)
            mel_spectrogram_data = extract_features_mel_spectrogram(data)
            
            if isinstance(mel_spectrogram_data, list):
                for idx, mel_data in enumerate(mel_spectrogram_data):
                    image_path = os.path.join(processed_data_dir, str(label), f"{i}_{idx}.png")
                    save_mel_spectrogram_image(mel_data, image_path)
            else:
                image_path = os.path.join(processed_data_dir, str(label), f"{i}.png")
                save_mel_spectrogram_image(mel_spectrogram_data, image_path)  
                
            i += 1