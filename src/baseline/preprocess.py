import os
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

data_dir = '/content/Data/genres_original'

X = []
y = []

for i, genre_folder in enumerate(os.listdir(data_dir)):
    genre_folder_path = os.path.join(data_dir, genre_folder)

    if os.path.isdir(genre_folder_path): 
        genre_files = os.listdir(genre_folder_path)

        for file_name in genre_files:
            if file_name.endswith('.wav'):
                file_path = os.path.join(genre_folder_path, file_name)
                features = extract_features_mfcc(file_path)
                X.append(features)
                y.append(i)


# extract_features_mfcc fails for one sample
# this is a quick workaround to remove this one sample
rem_indeces = []
for i, x in enumerate(X):
    if x is None:
        rem_indeces.append(i)

X.pop(rem_indeces[0])
y.pop(rem_indeces[0])

X_tensor = torch.from_numpy(np.array(X, dtype=np.float32))
y_tensor = torch.from_numpy(np.array(y, dtype=np.float32))

dataset = TensorDataset(X_tensor, y_tensor)

batch_size = len(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for data in dataloader:
    input, labels = data

    # Flatten mfccs 
    input = input.view(input.size(0), -1)
    
    predictions = KNN(input[:100], input, labels)
    print("Accuracy:", KNN_accuracy(predictions, labels[:100]))