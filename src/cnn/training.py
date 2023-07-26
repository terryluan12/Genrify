import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_model_name(name, batch_size, learning_rate, epoch):
    """
    Generate a name for the model consisting of all the hyperparameter values

    Args:
        name (str): name of the model
        batch_size (int): batch size
        learning_rate (float): learning rate
        epoch (int): number of epochs
    Returns:
        path (str): complete model name with path
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                    batch_size,
                                                    learning_rate,
                                                    epoch)
    return path

def train(model, train_loader, val_loader, num_epochs, learning_rate, batch_size):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loss_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []

    # confusion matrix for validation set
    confusion_matrix = torch.zeros(10, 10)

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = (correct / total) * 100
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Convert tensors to numpy arrays and move predicted to CPU
                labels_arr = labels.cpu().numpy()
                predicted_arr = predicted.detach().cpu().numpy()

                # Update confusion matrix
                for t, p in zip(labels_arr, predicted_arr):
                    confusion_matrix[t, p] += 1

        val_loss /= len(val_loader)
        val_accuracy = (val_correct / val_total) * 100
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        # save model
        model_path = get_model_name(model.name, batch_size, learning_rate, epoch)
        os.makedirs("./models", exist_ok=True)
        torch.save(model.state_dict(), f"./models/{model_path}.pt")
    
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]
    print("Training finished.")
    # model_path = get_model_name(model.name, batch_size, learning_rate, epoch)
    # os.makedirs("./csv", exist_ok=True)
    # np.savetxt(f"./csv/{model_path}_train_acc.csv", np.array(train_accuracy_list))
    # np.savetxt(f"./csv/{model_path}_val_acc.csv", np.array(val_accuracy_list))
    # np.savetxt(f"./csv/{model_path}_train_loss.csv", np.array(train_loss_list))
    # np.savetxt(f"./csv/{model_path}_val_loss.csv", np.array(val_loss_list))
    # np.savetxt(f"./csv/{model_path}_confusion_matrix.csv", confusion_matrix.numpy())
    