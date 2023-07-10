import librosa
import matplotlib.pyplot as plt

def get_spectrograms():

  audio_file = '/content/file_example_WAV_1MG.wav'
  audio, sample_rate = librosa.load(audio_file, sr=None)
  spectrogram = librosa.stft(audio)
  spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))

  plt.figure(figsize=(10, 4))
  librosa.display.specshow(spectrogram_db, sr=sample_rate, x_axis='time', y_axis='hz')
  plt.colorbar(format='%+2.0f dB')
  plt.title('Spectrogram')
  plt.show()
    
def convert_to_spectrogram_images(data_dir):
    """
    Converts the WAV files to Spectrogram features and saves them as images in a new directory structure.

    Args:
        data_dir (str): path to data directory
    """
    pass