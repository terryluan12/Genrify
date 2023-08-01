import torch
import torch.nn as nn

class MEL_CNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(MEL_CNN, self).__init__()
        self.name = "mel_cnn"

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)

        self.bn4 = nn.BatchNorm2d(128)

        
        self.fc_input_size = self.calculate_fc_input_size(dropout_rate)

        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 10)


    def calculate_fc_input_size(self, dropout_rate=0.25):
        # Perform a forward pass to get the output shape of the last convolutional layer
        with torch.no_grad():
            x = torch.zeros(1, 3, 500, 400)  # Example input size
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.conv4(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.dropout(x)

        return x.view(x.size(0), -1).shape[1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x