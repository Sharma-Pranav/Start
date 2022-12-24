import numpy as np
import torch
import torch.optim as optim
import cv2 
import pandas as pd
import sys
import os
sys.path.append("..\\..\\..\\src")

from datasets import CV2ImageDataset, dataset_loader
from segmentation_model import UNet
from model_class import NeuralNet
from run_phase import run_phase

import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SegmentationDataset(CV2ImageDataset):
    def __getitem__(self, idx):
        image = cv2.imread(self.paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.labels[idx], 0)/255 # Masking Level

        #print('mask : ',type(mask))
        #print(np.unique(mask))
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        #print(np.unique(mask))
        #print('check dimensions : ', image.shape, mask.shape)
        #image = image.to(self.device, dtype = torch.float)
        #mask = mask.to(self.device, dtype = torch.float)
        #print(torch.unique(mask, return_counts=True))
        return image, mask


def train_model(data_loader, model, device, lossfunction , optimizer, plot= False):
    from tqdm import tqdm 
    for X_train, y_train in tqdm(data_loader, desc='train'):
        X_train, y_train = X_train.to(device, dtype = torch.float), y_train.to(device, dtype = torch.float)
        #self.run_count +=1
        batch_size = X_train.shape[0]
        prediction = model(X_train)
        #print(type(loss))
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
        #print('prediction.shape, 512*512 : ', prediction.shape, 512*512)
        #print('unique : ', np.unique(prediction, return_counts = True))
        #print('batch_size : ', batch_size)
        batch_correct = (prediction == y_train).sum()
        #print('batch_correct : ', batch_correct)
        
        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].imshow(np.squeeze(prediction))
        axes[1].imshow(np.squeeze(y_train))
        for ax in axes:
            ax.axis('off')
        #plt.show()
        if plot:
            plt.savefig(os.path.join(os.getcwd(), 'plots', str(np.unique(batch_correct))+'.png'))

if __name__ == '__main__':    
    epochs = 12
    df = pd.read_csv('mapping_segmentation_labels.csv')
    train_df, test_df = train_test_split(df, test_size=0.33, random_state=42)
    
    image_height = 512
    image_width = 512
    aug = A.Compose([
    #A.CenterCrop(p=1, height=original_height, width=original_width),
    #A.RandomSizedCrop(min_max_height=(50, 101), height=original_height, width=original_width, p=1),
    A.VerticalFlip(p=0.5),     
    A.HorizontalFlip(p=0.5),          
    A.RandomRotate90(p=0.5), 
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
    #print(device)
    model.to(device)
    
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for i in range(epochs):
        print('epochs : ', i)
        train_model(trainloader, model, device, torch.nn.CrossEntropyLoss() , optimizer)
    train_model(testloader, model, device, torch.nn.CrossEntropyLoss() , optimizer, plot = True)
    #train_phase = run_phase(trainloader,  model, device, loss = torch.nn.CrossEntropyLoss(), optimizer= optimizer)
    
    #test_phase = run_phase(testloader,  model, device, loss = torch.nn.CrossEntropyLoss( ))
    
    #for epoch in range(epochs):
    #    train_phase.run()
    #    test_phase.run()
    
