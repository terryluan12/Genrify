import librosa
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
matplotlib.use('Agg')

def extract_features_mel_spectrogram(datapoint):
    """
    Extract mel-spectrogram features from audio file

    Args:
        datapoint (tuple(numpy.ndarray, int)):  Tuple of librosa sampled audio file
    Returns:
        mel_spec_db (np.array): Spectrogram features in dB (number of frequncy bins) x (number of time frames)
    """
    audio, sample_rate = datapoint
    mel_spectrogram = librosa.stft(audio) 
    mel_spec = librosa.feature.spectrogram(mel_spectrogram)
    mel_spec_db = librosa.amplitude_to_db(abs(mel_spec))

    return mel_spec_db

def get_mel_spectrograms():

  audio_file = '/content/file_example_WAV_1MG.wav'
  audio, sample_rate = librosa.load(audio_file, sr=None)
  mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
  mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

  plt.figure(figsize=(10, 4))
  librosa.display.specshow(mel_spectrogram_db, sr=sample_rate, x_axis='time', y_axis='mel')
  plt.colorbar(format='%+2.0f dB')
  plt.title('Mel-Spectrogram')
  plt.show()



def convert_to_mel_spectrogram_images(datasets, root_dir="."):
    """
    Converts the WAV files to Mel Spectrogram features and saves them as images in a new directory structure.

    Args:
        data_dir (str): path to data directory
        root_dir (string): path to the source directory of the Genrify module
    """
    print(f'Converting to Mel-Spectrogram')
    processed_data_dir = os.path.join(root_dir, "datasources/mel")
    os.makedirs(processed_data_dir, exist_ok=True)
    
    i = 0
    for dataset in datasets:
        for data, label in dataset:
            os.makedirs(os.path.join(processed_data_dir, str(label)), exist_ok=True)
            mel_spectrogram_data = extract_features_mel_spectrogram(data)
            image_path = os.path.join(processed_data_dir, str(label), f"{i}.png")
            save_spectrogram_image(mel_spectrogram_data, image_path)  
            
            i += 1