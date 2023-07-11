import librosa
import matplotlib.pyplot as plt
import os

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
        print("MFCC: Error encountered while parsing file: ", file_name)
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
    
def convert_to_mfcc_images(data_dir):
    """
    Converts the WAV files to MFCC features and saves them as images in a new directory structure.

    Args:
        data_dir (str): path to data directory
    """
    parent_dir = os.path.dirname(data_dir)
    processed_data_dir = os.path.join(parent_dir, 'mfcc')
    os.makedirs(processed_data_dir, exist_ok=True)

    for dirpath, dirnames, filenames in os.walk(data_dir):
        for dirname in dirnames:
            child_folder_path = os.path.join(dirpath, dirname)
            new_dir = os.path.join(processed_data_dir, dirname)
            os.makedirs(new_dir, exist_ok=True)
            for file in os.listdir(child_folder_path):
                # Do something with the file
                file_path = os.path.join(child_folder_path, file)
                mfccs = extract_features_mfcc(file_path)
                image_path = os.path.join(new_dir, f"{os.path.basename(file_path)}.png")
                save_mfcc_image(mfccs, image_path)

    print(f"Processed mfcc data saved in {processed_data_dir}")