import numpy as np
import torch
import torch.optim as optim
import cv2 
import pandas as pd
import sys
from Resnet import resnet18, resnet101
from Efficient_net import EfficientNet, efficient_net_config
sys.path.append("..\\..\\src")

from datasets import dataset_loader
from model_class import NeuralNet
from run_phase import run_phase
from pandas import DataFrame
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_height = 512
image_width = 512
contrast_brightness_values = [i*0.1 for i in range(-1, 4) if i!=0]
rgb_shift_values = [i*10 for i in range(-1, 3) if i!=0]

class HyperspectralDataset(Dataset):
    """
    Class for generation of Hyperspectral Dataset
    """
    def __init__(self, df:DataFrame, augmentation_list):
        self.df = df
        self.paths = df['path'].to_list()
        self.labels = df['label'].to_list()
        self.augmentation_list = augmentation_list
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        image = cv2.imread(self.paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = torch.tensor(self.labels[idx])#, 0)/255 # Masking Level
        image_list = []
        for i, transform in enumerate(self.augmentation_list):
            transformed = transform(image=image)#, mask=mask)
            transformed_image = transformed["image"]
            image_list.append(transformed_image)
        hyp_image = torch.vstack(image_list)
        return hyp_image, label

def get_datapoint(ds, device):
    """
    Run the respective training or testing phase
    Args:
        ds: dataset object
        device: Device to be used
    Returns: 
        X_train : X Datapoint
        y_train : Y Datapoint
    """
    for X_train, y_train in tqdm(ds):
        X_train, y_train = X_train.to(device, dtype = torch.float), y_train.to(device, dtype = torch.float)
        break
    return X_train, y_train
   
def get_augmentation_composition(specific_augmentation= None):
    """Get Augmentation Composition

    Args:
        specific_augmentation (Albumentation, optional): specific augmentation to be applied to compositions. Defaults to None.

    Returns:
        aug: Composition of augmentation
    """
    list_of_augmentation = []
    last_augmentation = [A.Resize(image_height, image_width), A.Normalize(), ToTensorV2()]
    if specific_augmentation:
        list_of_augmentation.append(specific_augmentation)
    list_of_augmentation.extend(last_augmentation)
    aug = A.Compose(list_of_augmentation)
    return aug

def get_all_compositions():
    """Get list of compositions

    Returns:
        list_of_compositions: list of compositions
    """
    list_of_compositions = []
    for elem in contrast_brightness_values:
        list_of_compositions.append(get_augmentation_composition(A.augmentations.transforms.RandomContrast(limit=[elem, elem], always_apply= True, p=1)))

    for elem in contrast_brightness_values:
        list_of_compositions.append(get_augmentation_composition(A.augmentations.transforms.RandomBrightness(limit=[elem, elem], always_apply= True, p=1)))
    
    for elem in rgb_shift_values:
        list_of_compositions.append(get_augmentation_composition(A.augmentations.transforms.RGBShift(r_shift_limit=[elem, elem] , g_shift_limit=0, b_shift_limit=0, always_apply=True, p=1)))

    for elem in rgb_shift_values:
        list_of_compositions.append(get_augmentation_composition(A.augmentations.transforms.RGBShift(r_shift_limit=0 , g_shift_limit=[elem, elem], b_shift_limit=0, always_apply=True, p=1)))

    for elem in rgb_shift_values:
        list_of_compositions.append(get_augmentation_composition(A.augmentations.transforms.RGBShift(r_shift_limit=0 , g_shift_limit=0, b_shift_limit=[elem, elem], always_apply=True, p=1)))    
    
    list_of_compositions.append(get_augmentation_composition())
    return list_of_compositions

class Net(nn.Module):
    """
    Simple Model Initialisation
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(54, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x =  x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TFNet(nn.Module):
    """
    Transfer Learning Model Initialisation
    """
    def __init__(self):
        super(TFNet, self).__init__()
        self.conv0 = nn.Conv2d(54, 3, 1)
        self.tl_model_weights = EfficientNet_V2_L_Weights.DEFAULT
        self.tl_model = efficientnet_v2_l(weights=self.tl_model_weights)
        self.fc1 = nn.Linear(1000, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv0(x)
        x = self.tl_model(x)
        x =  x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_efficient_net(version = 'b3', input_channels = 54, number_of_classes = 10):
    """Construct an efficient net from scratch

    Args:
        version (str, optional): The version of Efficient Net that one needs. Defaults to 'b3'.
        input_channels (int, optional): Number of input channels of the EfficientNet. Defaults to 54.
        number_of_classes (int, optional): Number of classes for the efficientnet. Defaults to 10.

    Returns:
        net: Efficient Net
    """
    width_mult, depth_mult, res, dropout_rate = efficient_net_config[version]
    net = EfficientNet(input_channels, width_mult, depth_mult, dropout_rate, num_classes = number_of_classes)
    return net
if __name__ == '__main__':    
    epochs = 10
    n_splits=5   
    image_height = 32
    image_width = 32
    batch_size = 64
    
    df = pd.read_csv('data\\train.csv')
    val_df = pd.read_csv('data\\test.csv')
    
    train_df, test_df = train_test_split(df, test_size=0.10, random_state=42)
    
    # Split into folds
    df["fold"] = np.nan
    skf = StratifiedKFold(n_splits=n_splits)
    skf.get_n_splits(df, df.label)
    for fold, (train_index, test_index) in enumerate(skf.split(df, df.label)):
        df.loc[test_index,"fold"]  = int(fold)

    early_stopping_length = 10

    best_model_per_fold = {}
    value_list = list(df.fold.unique())
    list_of_compositions = get_all_compositions()
    for fold in value_list:
        test_df =df.loc[df['fold'] == fold]
        train_df =df.loc[df['fold'] != fold]

        train_ds = HyperspectralDataset(train_df, list_of_compositions)
        test_ds = HyperspectralDataset(test_df, list_of_compositions)
        val_ds = HyperspectralDataset(val_df, list_of_compositions)

        train_ds_l = dataset_loader(train_ds, batch_size = batch_size)
        test_ds_l = dataset_loader(test_ds, batch_size = batch_size)
        val_ds_l = dataset_loader(val_ds, batch_size = batch_size)
        
        trainloader = train_ds_l.get_dataloader()
        testloader = test_ds_l.get_dataloader()
        valloader = val_ds_l.get_dataloader()
        
        train_ds_l.check_dataloader_dimension()
        test_ds_l.check_dataloader_dimension()
        val_ds_l.check_dataloader_dimension()

        #net = TFNet()
        #net = resnet101(54, 10)
        net = get_efficient_net(version = 'b7')
        nn_model = NeuralNet(net)
        model = nn_model.get_model()
        model.to(device)
        
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        
        train_phase = run_phase(trainloader,  model,'Train', device, loss = torch.nn.CrossEntropyLoss(), optimizer= optimizer)
        
        test_phase = run_phase(testloader,  model, 'Test', device, loss = torch.nn.CrossEntropyLoss( ))
        val_phase = run_phase(valloader,  model,'Validation', device, loss = torch.nn.CrossEntropyLoss( ))
        for epoch in range(epochs):
            train_accuracy_meter, train_loss_meter = train_phase.run()
            test_accuracy_meter, test_loss_meter = test_phase.run()
            val_accuracy_meter, val_loss_meter =  val_phase.run()
            
            continue_training = val_accuracy_meter.check_min_value_in_last_elements_of_queue(early_stopping_length)
            save_model_in_fold_flag = val_accuracy_meter.update_fold_on_min_flag()

            if not continue_training:
                break 
        torch.save(model, 'hyperspectral_model.pth')
        
        train_phase.accuracy_meter.plot(title='train_phase')
        test_phase.accuracy_meter.plot(title='test_phase')
        val_phase.accuracy_meter.plot(title='val_phase')    
        break

    #generate_plots(testloader, model, device, epochs, num_cols = 4, phase ='test')
