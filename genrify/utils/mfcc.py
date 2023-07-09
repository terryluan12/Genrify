import librosa
import matplotlib.pyplot as plt

def extract_features_mfcc(file_name):
    """
    Extract MFCC features from audio file

    Args:
        file_name (str): path to audio file
    Returns:
        mfccs (np.array): MFCC features
    """
    try:
        num_mfcc = 40
        audio, sample_rate = librosa.load(file_name)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=num_mfcc)
        mfccs = mfccs[:, 0:num_mfcc]
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None
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
    plt.figure(figsize=(10, 4), frameon=False)
    librosa.display.specshow(mfccs)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.cla()
    plt.close()