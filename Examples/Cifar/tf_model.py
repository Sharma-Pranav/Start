import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
#import torch.flatten as flatten



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.tl_model_weights = EfficientNet_V2_L_Weights.DEFAULT
        self.tl_model = efficientnet_v2_l(weights=self.tl_model_weights)
        self.fc1 = nn.Linear(1000, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.tl_model(x)
        x =  x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

