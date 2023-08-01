import torch.nn as nn
import torch.nn.functional as F

class ChromaClassifier(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(ChromaClassifier, self).__init__()
        self.name = "chroma_alex"
        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(-1, 256 * 6 * 6) #flatten feature data
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
