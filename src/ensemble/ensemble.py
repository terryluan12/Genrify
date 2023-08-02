from cnn import spectrogram_model, mfcc_model, chroma_model, mel_model
import numpy as np
import torch
import os

def get_weak_learners(dir="/content/drive/MyDrive/APS360 Team Project/Final Models"):
    spectrogram = spectrogram_model.Spectrogram_CNN()
    mfcc = mfcc_model.MFCC_Resnet()
    chroma = chroma_model.ChromaClassifier()
    mel_spectrogram  = mel_model.MEL_CNN()

    spectrogram_state = torch.load(os.path.join(dir, "spectrogram_model.pt"))
    mfcc_state = torch.load(os.path.join(dir, "mfcc_model.pt"))
    chroma_state = torch.load(os.path.join(dir, "chroma_model.pt"))
    mel_spectrogram_state = torch.load(os.path.join(dir, "mel_spectrogram_model.pt"))

    spectrogram.load_state_dict(spectrogram_state)
    mfcc.load_state_dict(mfcc_state)
    chroma.load_state_dict(chroma_state)
    mel_spectrogram.load_state_dict(mel_spectrogram_state)

    return spectrogram, mfcc, chroma, mel_spectrogram

def full_model(test_loaders, weak_learners=None, cuda=True, plot_dir="/content/Genrify/src/datasources"):
    if weak_learners is None:
        weak_learners = get_weak_learners()

    device = torch.device("cuda" if (torch.cuda.is_available() and cuda) else "cpu")

    for model in weak_learners:
        model.to(device)
        model.eval()
    
    correct = 0
    total = 0
    confusion_matrix = torch.zeros(10,10)
    
    with torch.no_grad():
        
        all_predictions = []
        all_labels = []
        for weak_learner_i, test_loader in enumerate(test_loaders):
            
            model_predictions = []
            model_labels = []
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Models have the same labels; no need to track each model's labels
                if weak_learner_i == 0:
                    model_labels.append(labels)

                outputs = weak_learners[weak_learner_i](inputs)
                _, predictions = torch.max(outputs.data, dim=1)
                model_predictions.append(predictions)

            # Same reason as above
            if weak_learner_i == 0:
                stacked_labels = torch.stack(model_labels, dim=1)
                all_labels.append(stacked_labels)

            stacked_model_predictions = torch.stack(model_predictions, dim=1)
            all_predictions.append(stacked_model_predictions)

        stacked_all_predictions = torch.stack(all_predictions, dim=2)
        majority_vote = torch.mode(stacked_all_predictions, dim=2).values

        for t, p in zip(all_labels[0].view(-1), majority_vote.view(-1)):
            confusion_matrix[t, p] += 1
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]
        np.savetxt(f"{plot_dir}/full_model_confusion_matirx.csv", confusion_matrix.numpy())

        correct_predictions = majority_vote == all_labels[0]
        
        correct = torch.sum(correct_predictions).item()
        total = all_labels[0].numel()

    accuracy = correct / float(total)
    return accuracy