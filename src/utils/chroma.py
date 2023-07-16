import librosa
import matplotlib.pyplot as plt
import os

def extract_features_chroma(datapoint):
    """
    Extract Chroma features from audio file

    Args:
        datapoint (tuple(numpy.ndarray, int)):  Tuple of librosa sampled audio file
    Returns:
        chroma (np.array): Chroma features
    """
    audio, sample_rate = datapoint
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    return chroma

def save_chroma_image(chroma, file_name):
    """
      Save Chroma features as an image

      Args:
          mfccs (np.array): Chroma features
          file_name (str): path to save image
    """
    if chroma is None:
          return
    plt.figure(figsize=(10, 4), frameon=False)
    librosa.display.specshow(chroma)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.cla()
    plt.close()
    
def convert_to_chroma_images(datasets, root_dir="."):
    """
    Converts the WAV files to Chroma features and saves them as images in a new directory structure.

    Args:
        datasets (List): list of datasets that are being converted to MFCC Images
        root_dir (string): path to the source directory of the Genrify module
    """
    print(f'Converting to Chroma')
    processed_data_dir = os.path.join(root_dir, "datasources/chroma")
    os.makedirs(processed_data_dir, exist_ok=True)

    i = 0
    for dataset in datasets:
        for data, label in dataset:
            os.makedirs(os.path.join(processed_data_dir, str(label)), exist_ok=True)
            chroma_features = extract_features_chroma(data)
            mfccs = extract_features_chroma(data)
            image_path = os.path.join(processed_data_dir, str(label), f"{i}.png")
            save_chroma_image(chroma_features, image_path)
            i += 1