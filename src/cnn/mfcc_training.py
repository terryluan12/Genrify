import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

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

def train(model, train_loader, val_loader, num_epochs, learning_rate, batch_size, step_size=20, gamma=0.5, patience=5):
    """
    train the model and save the best model. Uses early stopping and learning rate scheduler.

    Args:
        model (nn.Module): model to train
        train_loader (DataLoader): training data loader
        val_loader (DataLoader): validation data loader
        num_epochs (int): number of epochs
        learning_rate (float): learning rate
        batch_size (int): batch size
        step_size (int): step size for learning rate scheduler
        gamma (float): gamma for learning rate scheduler
        patience (int): patience for early stopping
    Returns:
        best_epoch (int): epoch with the best validation accuracy
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loss_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []

    
    best_val_accuracy = 0.0
    best_epoch = 0
    early_stopping_counter = 0

    # confusion matrix for validation set
    confusion_matrix = torch.zeros(10, 10)

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        model.train()
        
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
        
        scheduler.step()

        # Early stopping check
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            early_stopping_counter = 0
            # Save the best model checkpoint
            model_path = get_model_name(model.name, batch_size, learning_rate, epoch)
            os.makedirs("./models", exist_ok=True)
            torch.save(model.state_dict(), f"./models/{model_path}.pt")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered. Best validation accuracy: {best_val_accuracy:.2f}% at epoch {best_epoch}.")
                break

    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]
    print("Training finished.")
    model_path = get_model_name(model.name, batch_size, learning_rate, best_epoch)
    os.makedirs("./csv", exist_ok=True)
    np.savetxt(f"./csv/{model_path}_train_acc.csv", np.array(train_accuracy_list))
    np.savetxt(f"./csv/{model_path}_val_acc.csv", np.array(val_accuracy_list))
    np.savetxt(f"./csv/{model_path}_train_loss.csv", np.array(train_loss_list))
    np.savetxt(f"./csv/{model_path}_val_loss.csv", np.array(val_loss_list))
    np.savetxt(f"./csv/{model_path}_confusion_matrix.csv", confusion_matrix.numpy())
    return best_epoch

def test_model(model, test_loader, save_confusion_matrix=False, plot_dir="./csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    confusion_matrix = torch.zeros(10, 10)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # confusion matrix
            if save_confusion_matrix:
                # Convert tensors to numpy arrays and move predicted to CPU
                labels_arr = labels.cpu().numpy()
                predicted_arr = predicted.detach().cpu().numpy()

                # Update confusion matrix
                for t, p in zip(labels_arr, predicted_arr):
                    confusion_matrix[t, p] += 1

    test_accuracy = (correct / total) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    if save_confusion_matrix:
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]
        np.savetxt(f"{plot_dir}/{model.name}_confusion_matrix.csv", confusion_matrix.numpy())