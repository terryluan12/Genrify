import librosa
import matplotlib.pyplot as plt
import os

def extract_features_chroma(file_name):
    """
      Extract Chroma features from audio file

      Args:
          file_name (str): path to audio file
      Returns:
          chroma (np.array): Chroma features
      """
    try:
      audio, sample_rate = librosa.load(file_name)
      chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    except Exception as e:
      print("Chroma: Error encountered while parsing file: ", file_name)
      return None
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
    
def convert_to_chroma_images(data_dir):
    """
    Converts the WAV files to Chroma features and saves them as images in a new directory structure.

    Args:
        data_dir (str): path to data directory
    """
    parent_dir = os.path.dirname(data_dir)
    processed_data_dir = os.path.join(parent_dir, 'chroma')
    os.makedirs(processed_data_dir, exist_ok=True)

    for dirpath, dirnames, filenames in os.walk(data_dir):
        for dirname in dirnames:
            child_folder_path = os.path.join(dirpath, dirname)
            new_dir = os.path.join(processed_data_dir, dirname)
            os.makedirs(new_dir, exist_ok=True)
            for file in os.listdir(child_folder_path):
                file_path = os.path.join(child_folder_path, file)
                chroma_features = extract_features_chroma(file_path)
                image_path = os.path.join(new_dir, f"{os.path.basename(file_path)}.png")
                save_chroma_image(chroma_features, image_path)

    print(f"Processed chroma data saved in {processed_data_dir}")