import numpy as np
import torch
import torch.optim as optim
import pickle
import pandas as pd
import sys
import copy

sys.path.append("..\\..\\src")
#from glob import glob
#print("..\\..\\src : ", glob('..\\..\\src\\*'))
from datasets import CV2ImageDataset, dataset_loader
#print('imported dataset')
from cifar_model import Net
from model_class import NeuralNet
#print('imported NeuralNet')
from run_phase import run_phase
#print('imported run_phase')
import albumentations as A
from albumentations.pytorch import ToTensorV2

#from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
if __name__ == '__main__':    
    epochs = 2
    n_splits=2   
    image_height = 32
    image_width = 32
    batch_size = 64
    aug = A.Compose([
    #A.CenterCrop(p=1, height=original_height, width=original_width),
    #A.RandomSizedCrop(min_max_height=(50, 101), height=original_height, width=original_width, p=1),
    #A.VerticalFlip(p=0.5),     
    A.HorizontalFlip(p=0.5),          
    #A.RandomRotate90(p=0.5), 
    #A.Resize(image_height, image_width), 
    A.Normalize(),            
    ToTensorV2()])

    #print('glob() :', glob('data\\*'))
    df = pd.read_csv('data\\train.csv')
    val_df = pd.read_csv('data\\train.csv')
    # Split into folds

    df["fold"] = np.nan
    skf = StratifiedKFold(n_splits=n_splits)
    skf.get_n_splits(df, df.label)
    for fold, (train_index, test_index) in enumerate(skf.split(df, df.label)):
        df.loc[test_index,"fold"]  = int(fold)

    early_stopping_length = 10
    #print(df.fold.value_counts())
    #print(df.fold.unique())
    best_model_per_fold = {}
    value_list = list(df.fold.unique())
    #print('value_list: ', value_list)
    for fold in value_list:
        test_df =df.loc[df['fold'] == fold]
        train_df =df.loc[df['fold'] != fold]
        #print(train_df.shape, test_df.shape, fold)
        
        #a=b
        #train_df, test_df = train_test_split(df, test_size=0.33, random_state=42)
        #print('train_df.shape, test_df.shape : ', train_df.shape, test_df.shape)
        
        
        train_ds = CV2ImageDataset(train_df, transform=aug, device = device)
        test_ds = CV2ImageDataset(test_df, transform=aug, device = device)
        val_ds = CV2ImageDataset(val_df, transform=aug, device = device)
        
        train_ds_l = dataset_loader(train_ds, batch_size = batch_size)
        test_ds_l = dataset_loader(test_ds, batch_size = batch_size)
        val_ds_l = dataset_loader(val_ds, batch_size = batch_size)

        trainloader = train_ds_l.get_dataloader()
        testloader = test_ds_l.get_dataloader()
        valloader = val_ds_l.get_dataloader()

        #train_ds_l.check_dataloader_dimension()
        #test_ds_l.check_dataloader_dimension()
        #val_ds_l.check_dataloader_dimension()    
        net = Net()
        
        nn_model = NeuralNet(net)
        
        model = nn_model.get_model()
        #print(device)
        model.to(device)
        
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        
        train_phase = run_phase(trainloader,  model,'Train', device, loss = torch.nn.CrossEntropyLoss(), optimizer= optimizer)
        
        test_phase = run_phase(testloader,  model, 'Test', device, loss = torch.nn.CrossEntropyLoss( ))
        val_phase = run_phase(valloader,  model,'Validation', device, loss = torch.nn.CrossEntropyLoss( ))
        #print(train_phase)
        #print(test_phase)
        #print(val_phase)
        for epoch in range(epochs):
            train_accuracy_meter, train_loss_meter = train_phase.run()
            test_accuracy_meter, test_loss_meter = test_phase.run()
            val_accuracy_meter, val_loss_meter =  val_phase.run()
            
            continue_training = val_accuracy_meter.check_min_value_in_last_elements_of_queue(early_stopping_length)
            save_model_in_fold_flag = val_accuracy_meter.update_fold_on_min_flag()
            if save_model_in_fold_flag:
                best_model_per_fold[fold] = copy.deepcopy(nn_model)
            #print(continue_training)
            if not continue_training:
               #last_good_model = copy.deepcopy(nn_model)
               break 

        
        train_phase.accuracy_meter.plot(title='train_phase')
        test_phase.accuracy_meter.plot(title='test_phase')
        val_phase.accuracy_meter.plot(title='val_phase')

    with open('saved_cifar_model_dictionary.pkl', 'wb') as f:
        pickle.dump(best_model_per_fold, f)