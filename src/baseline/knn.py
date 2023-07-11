import torch

def KNN_accuracy(predictions, ground_truth):
    return ((predictions == ground_truth).sum().item() / len(ground_truth))

def KNN(test_mfccs, train_mfccs, train_labels):
    distances = torch.cdist(test_mfccs, train_mfccs)
    
    K = 5
    _, topk_indices = torch.topk(input=distances, k=K, largest=False, dim=1)
    
    nearest_labels = train_labels[topk_indices]
    
    predictions, _ = torch.mode(nearest_labels)
    
    return predictions