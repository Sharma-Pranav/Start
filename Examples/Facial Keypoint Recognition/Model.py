import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32,64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.Linear1 = nn.Linear(128* 25* 25, 512)
        self.Linear2 = nn.Linear(512, 256)
        self.Linear3 = nn.Linear(256, 136)
        self.drop = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.shape[0], -1)
        
        x = self.drop(F.relu(self.Linear1(x)))
        x = self.drop(F.relu(self.Linear2(x)))
        x = self.Linear3(x)
        
        return x