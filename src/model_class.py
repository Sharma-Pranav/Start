from torch.nn import Module
from torch.utils.data import DataLoader
class NeuralNet():
    """
    Class for inculcation of helper function for Neural Network
    """
    def __init__(self, neural_net:Module):
        """
        Initialisation of Neural Network
        Args:
            neural_net : Neural Network
        """
        self.model = neural_net
        
    def get_model(self):
        """
        Get the model from the class
        Returns:
            self.model : Neural Network
        """
        return self.model

    def model_dimension_check(self, data_loader: DataLoader):
        """
        Model dimension print
        Args:
            data_loader : Dataloader
        """
        for i, (inputs, labels) in enumerate(data_loader):
            outputs = self.model(inputs)
            print('Shape of Outputs : ', outputs.shape)
            break

    def freeze_model(self):
        """
        Freeze Model layers
        """
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_model(self):
        """
        Unfreeze Model layers
        """
        for param in self.model.parameters():
            param.requires_grad = True
