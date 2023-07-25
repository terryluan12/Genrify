import torch
import torch.nn as nn

class Chroma_CNN(nn.Module):
    def __init__(self, kernelSizes = [3, 3, 3, 3], stride=1, padding=0):
        super(Chroma_CNN, self).__init__()
        self.name = "chroma_cnn"

        self.conv1 = nn.Conv2d(3, 15, kernel_size=kernelSizes[0], stride=stride, padding=padding)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(15, 30, kernel_size=kernelSizes[1], stride=stride, padding=padding)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(30, 60, kernel_size=kernelSizes[2], stride=stride, padding=padding)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(60, 120, kernel_size=kernelSizes[3], stride=stride, padding=padding)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the input tensor to the fully connected layer
        self.fc_input_size = self.calculate_fc_input_size()

        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def calculate_fc_input_size(self):
        # Perform a forward pass to get the output shape of the last convolutional layer
        with torch.no_grad():
            x = torch.zeros(1, 3, 1000, 400)  # Example input size
            x = self.conv1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.maxpool2(x)
            x = self.conv3(x)
            x = self.maxpool3(x)
            x = self.conv4(x)
            x = self.maxpool4(x)

        return x.view(x.size(0), -1).shape[1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)

        return x