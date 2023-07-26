from cnn import spectrogram_model, mfcc_model #chroma_model, mel_spectrogram_model
import torch
import os

def get_models(dir="/content/drive/MyDrive/APS360 Team Project/Final Models"):
    spectrogram = spectrogram_model.Spectrogram_CNN()
    # mfcc = mfcc_model.MFCC_CNN()
    # chroma = chroma_model.Chroma_CNN()
    # mel_spectrogram  = mel_spectrogram_model.Mel_Spectrogram_CNN()

    spectrogram_state = torch.load(os.path.join(dir, "spectrogram_model.pt"))
    # mfcc_state = torch.load(os.path.join(dir, "mfcc_model.pt"))
    # chroma_state = torch.load(os.path.join(dir, "chroma_model.pt"))
    # mel_spectrogram_state = torch.load(os.path.join(dir, "mel_spectrogram_model.pt"))

    spectrogram.load_state_dict(spectrogram_state)
    # mfcc.load_state_dict(mfcc_state)
    # chroma.load_state_dict(chroma_state)
    # mel_spectrogram.load_state_dict(mel_spectrogram_state)


    return spectrogram, None, None, None
    # return spectrogram, mfcc, chroma_model, mel_spectrogram


