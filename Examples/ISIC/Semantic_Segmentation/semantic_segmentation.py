import numpy as np
import torch
import torch.optim as optim
import cv2 
import pandas as pd
import sys
import os
from tqdm import tqdm
sys.path.append("..\\..\\..\\src")

from datasets import CV2ImageDataset, dataset_loader
from segmentation_model import UNet
from model_class import NeuralNet
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SegmentationDataset(CV2ImageDataset):
    """
    Class for generation of Segmentation Dataset
    """
    def __getitem__(self, idx):
        image = cv2.imread(self.paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.labels[idx], 0)/255 # Masking Level
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]#
        return image, mask


def run_phase(data_loader, model, device, lossfunction , optimizer = None):
    """
    Run the respective training or testing phase
    Args:
        data_loader: Dataloader object
        model: Neural Network Model
        device: Device to be used
        lossfunction: Loss Function
        optimizer: Optimizer
    Returns: 
        loss: Averaged Loss
    """
    loss_count = 0
    for X_train, y_train in tqdm(data_loader, desc='train'):
        X_train, y_train = X_train.to(device, dtype = torch.float), y_train.to(device, dtype = torch.float)
        prediction = model(X_train)
        loss = lossfunction(prediction,y_train.to(torch.long))
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = loss.cpu().detach().numpy()

        X_train = X_train.cpu().detach().numpy()
        y_train = y_train.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()
        prediction = np.argmax(prediction, axis =1)
        loss_count += loss    
    return loss_count/len(data_loader)

def generate_plots(data_loader, model, device, epoch, num_cols = 4, phase ='test'):
    """
    Run the respective training or testing phase
    Args:
        data_loader: Dataloader object
        model: Neural Network Model
        device: Device to be used
        epoch: Epoch
        num_cols: number of columns to be plotted
        phase: Respective phase
    """
    i = 0
    fig, axes = plt.subplots(nrows=num_cols, ncols=2)
    for X_train, y_train in tqdm(data_loader, desc=phase):
        X_train, y_train = X_train.to(device, dtype = torch.float), y_train.to(device, dtype = torch.float)
        prediction = model(X_train)
        X_train = X_train.cpu().detach().numpy()
        y_train = y_train.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()
        prediction = np.argmax(prediction, axis =1)
        
        axes[i][0].imshow(np.squeeze(prediction), cmap = 'gray')
        axes[i][1].imshow(np.squeeze(y_train), cmap = 'gray')
        for ax in axes[i]:
            ax.axis('off')
        i +=1 
        if i == num_cols:
            break
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()        
    plt.savefig(os.path.join(os.getcwd(), 'plots', phase + '_'+ str(epoch) + '_'+ str(i) +'.jpg'))


if __name__ == '__main__':    
    epochs = 15
    df = pd.read_csv('mapping_segmentation_labels.csv')
    train_df, test_df = train_test_split(df, test_size=0.33, random_state=42)
    
    image_height = 512
    image_width = 512
    aug = A.Compose([
    #A.VerticalFlip(p=0.5),     
    #A.HorizontalFlip(p=0.5),          
    #A.RandomRotate90(p=0.5), 
    A.Resize(image_height, image_width), 
    A.Normalize(),            
    ToTensorV2()])
    
    train_ds = SegmentationDataset(train_df, transform=aug, device = device)
    test_ds = SegmentationDataset(test_df, transform=aug, device = device)
    
    train_ds_l = dataset_loader(train_ds)
    test_ds_l = dataset_loader(test_ds)
    
    trainloader = train_ds_l.get_dataloader()
    testloader = test_ds_l.get_dataloader()
    
    train_ds_l.check_dataloader_dimension()
    test_ds_l.check_dataloader_dimension()
    
    net = UNet()
    nn_model = NeuralNet(net)
    model = nn_model.get_model()
    model.to(device)
    
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    for i in range(epochs):
        print('epochs : ', i)
        loss = run_phase(trainloader, model, device, torch.nn.CrossEntropyLoss() , optimizer)
        
        print('Train loss : ', loss)
    loss = run_phase(testloader, model, device, torch.nn.CrossEntropyLoss())
    print('Test loss : ', loss)
    
    generate_plots(testloader, model, device, epochs, num_cols = 4, phase ='test')