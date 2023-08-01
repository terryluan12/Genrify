import librosa
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')

def extract_features_mfcc(datapoint):
    """
    Extract MFCC features from audio file

    Args:
        datapoint (tuple(numpy.ndarray, int)):  Tuple of librosa sampled audio file
    Returns:
        mfccs (np.array): MFCC features
    """
    num_mfcc = 40
    audio, sample_rate = datapoint
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
    
def convert_to_mfcc_images(datasets, root_dir="."):
    """
    Converts the WAV files to MFCC features and saves them as images in a new directory structure.

    Args:
        datasets (List): list of datasets that are being converted to MFCC Images
        root_dir (string): path to the source directory of the Genrify module
    """
    print(f'Converting to MFCC')
    processed_data_dir = os.path.join(root_dir, "datasources/mfcc")
    os.makedirs(processed_data_dir, exist_ok=True)
    
    i = 0
    for dataset in datasets:
        for data, label in dataset:
            os.makedirs(os.path.join(processed_data_dir, str(label)), exist_ok=True)
            mfccs = extract_features_mfcc(data)
            image_path = os.path.join(processed_data_dir, str(label), f"{i}.png")
            save_mfcc_image(mfccs, image_path)
            i += 1