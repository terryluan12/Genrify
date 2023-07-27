from cnn import spectrogram_model, mfcc_model #chroma_model, mel_spectrogram_model
import torch
import os

def get_weak_learners(dir="/content/drive/MyDrive/APS360 Team Project/Final Models"):
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


    return spectrogram, spectrogram, spectrogram, spectrogram
    # return spectrogram, mfcc, chroma_model, mel_spectrogram

def full_model(data_loader, cuda=True, weak_learners=None):
    if weak_learners is None:
        weak_learners = get_weak_learners()

    device = torch.device("cuda" if (torch.cuda.is_available() and cuda) else "cpu")

    for model in weak_learners:
        model.to(device)
        model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            all_predictions = []
            for model in weak_learners:
                outputs = model(inputs)
                _, predictions = torch.max(outputs.data, dim=1)
                all_predictions.append(predictions)

            stacked_predictions = torch.stack(all_predictions, dim=1)

            majority_vote = torch.mode(stacked_predictions, dim=1).values
            print(labels.size())
            correct += (majority_vote == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / float(total)
    return accuracy





    
    


