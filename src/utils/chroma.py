import librosa
import matplotlib.pyplot as plt
import os
import matplotlib
from utils.augment_sample import augment_sample
matplotlib.use('Agg')

def extract_features_chroma(datapoint, training=False):
    """
    Extract Chroma features from an audio file.

    Args:
        datapoint (tuple(numpy.ndarray, int)): A tuple containing the audio data as a numpy array and the sample rate as an integer.
        training (bool): Whether the extraction is for training data or not.

    Returns:
        (np.array or list of np.array): Chroma features. If training is True, returns a list of Chroma features where each item is of shape (number of Chroma bins) x (number of time frames). If training is False, returns a single Chroma feature of shape (number of Chroma bins) x (number of time frames).
    """
    audio, sample_rate = datapoint
    if training:
        augmented_chromas = []
        audios = augment_sample(audio, sample_rate)
        for audio in audios:
            augmented_chromas.append(librosa.feature.chroma_cqt(y=audio, sr=sample_rate, n_chroma=48, bins_per_octave=48))
        return augmented_chromas
    else:
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate, n_chroma=48, bins_per_octave=48)
        return chroma

def save_chroma_image(chroma, file_name):
    """
      Save Chroma features as an image

      Args:
          chroma (np.array): Chroma features
          file_name (str): path to save image
    """
    if chroma is None:
          return
    plt.figure(figsize=(2.24, 2.24), dpi=100, frameon=False)
    librosa.display.specshow(chroma)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.cla()
    plt.close('all')
    
def convert_to_chroma_images(datasets, root_dir=".", training=False):
    """
    Converts the WAV files to Chroma features and saves them as images in a new directory structure.

    Args:
        datasets (List): List of datasets that are being converted to Chroma images.
        root_dir (string): Path to the source directory of the Genrify module.
        training (bool): Whether the conversion is for training or not.
    """
    print(f'Converting to Chroma')
    processed_data_dir = os.path.join(root_dir, "datasources/chroma")
    os.makedirs(processed_data_dir, exist_ok=True)

    i = 0
    for dataset in datasets:
        for data, label in dataset:
            os.makedirs(os.path.join(processed_data_dir, str(label)), exist_ok=True)
            chroma_features = extract_features_chroma(data, training=training)
            if isinstance(chroma_features, list):
                for idx, chroma_data in enumerate(chroma_features):
                    image_path = os.path.join(processed_data_dir, str(label), f"{i}_{idx}.png")
                    save_chroma_image(chroma_data, image_path)
            else:
                image_path = os.path.join(processed_data_dir, str(label), f"{i}.png")
                save_chroma_image(chroma_features, image_path)
            i += 1
