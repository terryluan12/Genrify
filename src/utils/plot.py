import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation accuracy/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    train_acc = np.loadtxt("./csv/{}_train_acc.csv".format(path))
    val_acc = np.loadtxt("./csv/{}_val_acc.csv".format(path))
    train_loss = np.loadtxt("./csv/{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("./csv/{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Accuracy")
    n = len(train_acc) # number of epochs
    plt.plot(range(1,n+1), train_acc, label="Train")
    plt.plot(range(1,n+1), val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

def plot_confusion_matrix(model_path, classes):
    confusion_matrix = np.loadtxt(f"./csv/{model_path}_confusion_matrix.csv")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()