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

def full_model(data_loader, weak_learner_1=None, weak_learner_2=None, weak_learner_3=None, weak_learner_4=None):
    
    if not weak_learner_1:
        weak_learner_1, weak_learner_2, weak_learner_3, weak_learner_4 = get_weak_learners()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        weak_learner_1.to(device)
        weak_learner_1.eval()
        outputs1 = weak_learner_1(inputs)
        _, pred1 = torch.max(outputs1.data, dim=1)

        weak_learner_2.to(device)
        weak_learner_2.eval()
        outputs2 = weak_learner_2(inputs)
        _, pred2 = torch.max(outputs2.data, dim=1)

        weak_learner_3.to(device)
        weak_learner_3.eval()
        outputs3 = weak_learner_3(inputs)
        _, pred3 = torch.max(outputs3.data, dim=1)

        weak_learner_4.to(device)
        weak_learner_4.eval()
        outputs4 = weak_learner_4(inputs)
        _, pred4 = torch.max(outputs4.data, dim=1)

        stacked_predictions = torch.stack([pred1, pred2, pred3, pred4], dim=1)

        majority_vote, _ = torch.mode(stacked_predictions, dim=0)

        correct += (majority_vote == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / float(total)
    
    return accuracy




    
    


