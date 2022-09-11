from typing import Any, Optional
import numpy as np

from torch.utils.data.dataloader import DataLoader
import torch
from torch.nn import Module
from torch.optim import Optimizer
from tqdm import tqdm
from average_meter import AverageMeter
class run_phase():
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
        #print(model)
        #a=b
        self.device = device
        self.optimizer = optimizer
        self.epoch = 0
        #if optimizer:
        #    self.optimizer = optimizer
        self.compute_loss = loss
        self.accuracy_meter = AverageMeter()
        self.loss_meter = AverageMeter()
        self.phase = phase#'Validation' if optimizer is None else 'Training'
    def update_epoch(self):
        self.epoch+=1
    def run(self):
        
        for X_train, y_train in tqdm(self.loader, desc=self.phase):
            X_train, y_train = X_train.to(self.device, dtype = torch.float), y_train.to(self.device, dtype = torch.float)
            self._run_single(X_train, y_train)
        self.update_epoch()
        print('{} Accuracy for epoch : {}'.format(self.phase, self.accuracy_meter.return_current_avg()))    
        print('{} Loss for epoch : {}'.format(self.phase, self.loss_meter.return_current_avg())) 
        return self.accuracy_meter, self.loss_meter
        #self.accuracy_meter.reset()
    def run_for_epoch(self):
        if self.optimizer==None:
            with torch.no_grad():
                self.run()
        else:
            self.run()
            
        
        #log(self.avg)
    
    def _run_single(self, X_train, y_train):
        self.run_count +=1
        batch_size = X_train.shape[0]
        prediction = self.model(X_train)
        #print('prediction.shape, y_train.shape : ', prediction.shape, y_train.shape)
        #print('np.unique(prediction), np.unique(y_train) : ', torch.unique(prediction), torch.unique(y_train))
        #a=b
        loss = self.compute_loss(prediction,y_train.to(torch.long))
        if self.optimizer:
            #print('inside optimizer ')
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        loss = loss.cpu().detach().numpy()

        X_train = X_train.cpu().detach().numpy()
        y_train = y_train.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()
        prediction = np.argmax(prediction, axis =1)
        #print(prediction)
        #y_numpy = y_train.cpu().detach().numpy()
        #y_prediction_np = np.argmax(prediction, axis =1)
        #print('y_prediction_np: ', np.unique(prediction))
        #print('y_train : ', np.unique(y_train))
        #a=b
        #print('prediction.shape[1]*prediction.shape[2] : ', prediction.shape[1]*prediction.shape[2])
        #prediction[prediction>0.5]=1
        #prediction.shape, y_train.shape
        batch_correct = (prediction == y_train).sum()#/(prediction.shape[1]*prediction.shape[2])#/batch_size
        #print(batch_correct, prediction.shape[0])
        #print('batch_correct, batch_size : ', batch_correct, batch_size)
        self.accuracy_meter.update(batch_correct, batch_size, self.epoch)
        self.loss_meter.update(loss, batch_size, self.epoch)
        
        
            
        
            
    