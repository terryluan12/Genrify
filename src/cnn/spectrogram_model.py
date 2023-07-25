import torch
import torch.nn as nn

class Spectrogram_CNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(Spectrogram_CNN, self).__init__()
        self.name = "3_layer_spec"

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the input tensor to the fully connected layer
        self.fc_input_size = self.calculate_fc_input_size()

        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.relu4 = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout_rate)  # Add dropout to fully connected layers
        self.fc2 = nn.Linear(512, 10)

        self.dropout_conv = nn.Dropout2d(dropout_rate)  # Add dropout to convolutional layers

    def calculate_fc_input_size(self):
        # Perform a forward pass to get the output shape of the last convolutional layer
        with torch.no_grad():
            x = torch.zeros(1, 3, 500, 400)  # Example input size
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.maxpool2(x)
            x = self.conv3(x)
            x = self.relu3(x)
            x = self.maxpool3(x)

        return x.view(x.size(0), -1).shape[1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.dropout_conv(x)  # Apply dropout to the first convolutional layer

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout_conv(x)  # Apply dropout to the second convolutional layer

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.dropout_conv(x)  # Apply dropout to the third convolutional layer

        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout_fc(x)  # Apply dropout to fully connected layer
        x = self.fc2(x)

        return x
