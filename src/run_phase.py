from typing import Any
import numpy as np

from torch.utils.data.dataloader import DataLoader
import torch
from torch.nn import Module
from tqdm import tqdm
from average_meter import AverageMeter
class run_phase():
    """
    Phase for running class
    """
    def __init__(self, loader: DataLoader[Any],
                 model:Module, 
                 phase:str,
                 device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 loss = torch.nn.CrossEntropyLoss(),
                 optimizer: torch.optim = None,
                 ):
        self.run_count = 0
        self.loader = loader
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.epoch = 0
        self.compute_loss = loss
        self.accuracy_meter = AverageMeter()
        self.loss_meter = AverageMeter()
        self.phase = phase

    def update_epoch(self):
        """
        Update Epochs
        """
        self.epoch+=1
    def run(self):
        """
        Run for number of epochs

        Returns:
           self.accuracy_meter:  Accuracy Meter Object, 
           self.loss_meter : Loss Meter Object
        """

        for X_train, y_train in tqdm(self.loader, desc=self.phase):
            X_train, y_train = X_train.to(self.device, dtype = torch.float), y_train.to(self.device, dtype = torch.float)
            self._run_single(X_train, y_train)
        self.update_epoch()
        print('{} Accuracy for epoch : {}'.format(self.phase, self.accuracy_meter.return_current_avg()))    
        print('{} Loss for epoch : {}'.format(self.phase, self.loss_meter.return_current_avg())) 
        return self.accuracy_meter, self.loss_meter

    def run_for_epoch(self):
        """
        Run for Epochs
        """
        if self.optimizer==None:
            with torch.no_grad():
                self.run()
        else:
            self.run()
        
    def _run_single(self, X_train, y_train):
        """
        Run for single epoch
        Returns:
            X_train: data
            y_train: labels
        """
        
        self.run_count +=1
        batch_size = X_train.shape[0]
        prediction = self.model(X_train)
        loss = self.compute_loss(prediction,y_train.to(torch.long))
        if self.optimizer:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        loss = loss.cpu().detach().numpy()

        X_train = X_train.cpu().detach().numpy()
        y_train = y_train.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()
        prediction = np.argmax(prediction, axis =1)
        batch_correct = (prediction == y_train).sum()
        self.accuracy_meter.update(batch_correct, batch_size, self.epoch)
        self.loss_meter.update(loss, batch_size, self.epoch)
        
        
            
        
            
    