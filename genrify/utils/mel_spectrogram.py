import librosa
import matplotlib.pyplot as plt

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



def convert_to_mel_spectrogram_images(data_dir):
    """
    Converts the WAV files to Mel Spectrogram features and saves them as images in a new directory structure.

    Args:
        data_dir (str): path to data directory
    """
    pass
