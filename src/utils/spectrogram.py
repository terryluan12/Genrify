import librosa
import matplotlib.pyplot as plt

def extract_features_spectrogram(file_name):
    """
    Extract spectrogram features from audio file

    Args:
        file_name (str): path to audio file
    Returns:
        spectrogram_db (np.array): Spectrogram features in dB (number of frequncy bins) x (number of time frames)
    """
    try:
        audio, sample_rate = librosa.load(file_name, sr=22000) # Want to plot up to 11kHz -> Requires 22kHz sampling freq
        spectrogram = librosa.stft(audio) 
        spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None
    return spectrogram_db

def save_spectrogram_image(spectrogram_data, file_name):
    """
    Save spectrogram data as an image

    Args:
        spectrogram_data (np.array): Spectrogram features in dB (number of frequncy bins) x (number of time frames) 
        file_name (str): path to save image
    """
    if spectrogram_data is None:
        return
    plt.figure(figsize=(10, 4), frameon=False)
    plt.ylim(0, 11000) # Always plot up to 11kHz
    librosa.display.specshow(spectrogram_data, x_axis='time', y_axis='hz', vmin=-40, vmax=40) # Always show magnitude from -40 to +40 dB
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.cla()
    plt.close()  