from torch.nn import Module
from torch.utils.data import DataLoader
class NeuralNet():
    def __init__(self, neural_net:Module):
        self.model = neural_net
        
    def get_model(self):
        return self.model

    def model_dimension_check(self, data_loader: DataLoader):
        for i, (inputs, labels) in enumerate(data_loader):
            outputs = self.model(inputs)
            print('Shape of Outputs : ', outputs.shape)
            break

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = True
