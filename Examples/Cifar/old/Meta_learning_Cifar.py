import numpy as np
import torch
import torch.optim as optim
import pickle
import pandas as pd
import sys
import copy

sys.path.append("..\\..\\src")

from datasets import CV2ImageDataset, dataset_loader
from tf_model import Net
#from cifar_model import Net
from model_class import NeuralNet
from run_phase import run_phase
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
import pickle

from meta_learning_tools import MetaLearn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def model_prototype():
    net = Net()
    return net#model_prototype

with open('saved_cifar_model_weights_dictionary_with_tl.pkl', 'rb') as fp:
    model_weight_dict = pickle.load(fp)
#print(model_weight_dict) 
model_list = []
for key in model_weight_dict.keys():
    #print(model_weight_dict[key].model)
    model_weight = model_weight_dict[key]
    model = model_prototype()
    model.load_state_dict(model_weight)
    model.cpu()
    #print(type(model))
    #print(model)
    model.eval()
    #rand_input = torch.rand((1,3, 32,32))
    #rand_out = model(rand_input)
    #print(rand_input.shape)
    model_list.append(model)

#print(model_list)
#a=b
train_df = pd.read_csv('data\\train.csv')
val_df = pd.read_csv('data\\test.csv')


meta_learn = MetaLearn(model_list)
tta_transforms = [
    A.augmentations.transforms.ChannelShuffle(p=0.5),
    A.HorizontalFlip(p=0.5), 
]

train_tta_dict= meta_learn.generate_tta_dict_for_folds(train_df , tta_transforms, device)
val_tta_dict= meta_learn.generate_tta_dict_for_folds(val_df , tta_transforms, device)

from datasets import CV2ImageDataset, dataset_loader
from run_phase import run_phase
batch_size = 64
aug = A.Compose([ 
    A.Normalize(),            
    ToTensorV2()])
val_ds = CV2ImageDataset(val_df, transform=aug, device = device)
val_ds_l = dataset_loader(val_ds, batch_size = batch_size)
valloader = val_ds_l.get_dataloader()
# Just Sanity Check
for i, model in enumerate(model_list):
    print('model : ', i)
    val_phase = run_phase(valloader, model.to(device),'Validation', device, loss = torch.nn.CrossEntropyLoss( ))
    val_accuracy_meter, val_loss_meter =  val_phase.run()

train_tta_df=meta_learn.create_meta_learn_labels_on_dict(train_tta_dict)
val_tta_df=meta_learn.create_meta_learn_labels_on_dict(val_tta_dict)
print( val_tta_df.shape,train_tta_df.shape)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

val_acc, val_log_loss = meta_learn.do_meta_learning_on_tta_dicts(RandomForestClassifier(), train_tta_df, val_tta_df, 'custom')

print(val_acc, val_log_loss)