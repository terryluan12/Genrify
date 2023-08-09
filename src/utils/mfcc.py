import librosa
import matplotlib.pyplot as plt
import os
import matplotlib
from utils.augment_sample import augment_sample
matplotlib.use('Agg')

def extract_features_mfcc(datapoint, training=False):
    """
    Extract MFCC features from an audio file.

    Args:
        datapoint (tuple(numpy.ndarray, int)): A tuple containing the audio data as a numpy array and the sample rate as an integer.
        training (bool): Whether the extraction is for training data or not.

    Returns:
        (np.array or list of np.array): MFCC features. If training is True, returns a list of MFCC features, where each item is of shape (number of MFCC coefficients) x (number of time frames). If training is False, returns a single MFCC feature of shape (number of MFCC coefficients) x (number of time frames).
    """
    num_mfcc = 40
    audio, sample_rate = datapoint
    if training:
        augmented_mfccs = []
        audios = augment_sample(audio, sample_rate)
        for audio in audios:
            augmented_mfccs.append(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=num_mfcc))
        return augmented_mfccs
    else:
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=num_mfcc)
        return mfccs

# save mfcc features to an image
def save_mfcc_image(mfccs, file_name):
    """
    Save MFCC features as an image

    Args:
        mfccs (np.array): MFCC features
        file_name (str): path to save image
    """
    if mfccs is None:
        return
    plt.figure(figsize=(3, 2), frameon=False)
    librosa.display.specshow(mfccs)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.cla()
    plt.close('all')
    
def convert_to_mfcc_images(dataset, root_dir=".", training=False):
    """
    Converts the WAV files to MFCC features and saves them as images in a new directory structure.

    Args:
        dataset (torch.utils.data.Subset): Subset of the original dataset containing audio data and corresponding labels.
        root_dir (string): Path to the source directory of the Genrify module.
        training (bool): Whether the conversion is for training or not.
    """
    print(f'Converting to MFCC')
    processed_data_dir = os.path.join(root_dir, "datasources/mfcc")
    os.makedirs(processed_data_dir, exist_ok=True)
    
    i = 0
    for data, label in dataset:
        os.makedirs(os.path.join(processed_data_dir, str(label)), exist_ok=True)
        mfccs = extract_features_mfcc(data, training=training)
        if isinstance(mfccs, list):
            for idx, mfcc_data in enumerate(mfccs):
                image_path = os.path.join(processed_data_dir, str(label), f"{i}_{idx}.png")
                save_mfcc_image(mfcc_data, image_path)
        else:
            image_path = os.path.join(processed_data_dir, str(label), f"{i}.png")
            save_mfcc_image(mfccs, image_path)
        i += 1
