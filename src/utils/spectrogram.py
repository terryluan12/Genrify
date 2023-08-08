import librosa
import matplotlib.pyplot as plt
import os
import matplotlib
from augment_sample import augment_sample

matplotlib.use('Agg')

def extract_features_spectrogram(datapoint, training=False):
    """
    Extract spectrogram features from an audio file.

    Args:
        datapoint (tuple(numpy.ndarray, int)): A tuple containing the audio data as a numpy array and the sample rate as an integer.
        training (bool): Whether the extraction is for training data or not.

    Returns:
        (np.array or list of np.array): Spectrogram features in dB. If training is True, returns a list of spectrogram features where each item is of shape (number of frequency bins) x (number of time frames). If training is False, returns a single spectrogram feature of shape (number of frequency bins) x (number of time frames).
    """
    audio, sample_rate = datapoint
    if training:
        augmented_spectrograms = []
        audios = augment_sample(audio, sample_rate)
        for audio in audios:
            spectrogram = librosa.stft(audio)
            spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
            augmented_spectrograms.append(spectrogram_db)
        return augmented_spectrograms
    else: 
        spectrogram = librosa.stft(audio) 
        spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))

        return spectrogram_db

def save_spectrogram_image(spectrogram_data, file_name):
    """
    Save spectrogram data as an image

    Args:
        spectrogram_data (np.array): Spectrogram features in dB (number of frequncy bins) x (number of time frames) 
        file_name (str): path to save image
    """
    if spectrogram_data is None:
        return
    plt.figure(figsize=(5, 4), frameon=False)
    plt.ylim(0, 11000) # Always plot up to 11kHz
    librosa.display.specshow(spectrogram_data, x_axis='time', y_axis='hz', vmin=-40, vmax=40) # Always show magnitude from -40 to +40 dB
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.cla()
    plt.close('all')

def convert_to_spectrogram_images(datasets, root_dir=".", training=False):
    """
    Converts the WAV files to Spectrogram features and saves them as images in a new directory structure.

    Args:
        datasets (List): List of datasets that are being converted to Spectrogram images.
        root_dir (string): Path to the source directory of the Genrify module.
        training (bool): Whether the conversion is for training or not.
    """
    print(f'Converting to Spectrogram')
    processed_data_dir = os.path.join(root_dir, "datasources/spectrogram")
    os.makedirs(processed_data_dir, exist_ok=True)
    
    i = 0
    for dataset in datasets:
        for data, label in dataset:
            os.makedirs(os.path.join(processed_data_dir, str(label)), exist_ok=True)
            spectrogram_data = extract_features_spectrogram(data, training=training)
            if isinstance(spectrogram_data, list):
                for idx, spec_data in enumerate(spectrogram_data):
                    image_path = os.path.join(processed_data_dir, str(label), f"{i}_{idx}.png")
                    save_spectrogram_image(spec_data, image_path)
            else:
                image_path = os.path.join(processed_data_dir, str(label), f"{i}.png")
                save_spectrogram_image(spectrogram_data, image_path)
            
            i += 1