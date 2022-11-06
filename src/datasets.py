
from numpy import array
from pandas import DataFrame
from torchvision import transforms
import os
import torch
from PIL import Image
import cv2 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, TensorDataset, DataLoader


def get_from_df_paths_targets( df: DataFrame, transform=None):
    """
    Get from dataframe paths and labels
    Args:
        df: Dataframe
        transform: transform
    Returns:
        df: Dataframe
        paths: paths
        labels: labels
        transform: transform
    """
    paths = df['path'].to_list()
    labels = df['label'].to_list()
    return df,paths, labels, transform

def display_image_grid(images_filepaths:list, predicted_labels=(), cols=5):
    """
    Displays images in grid 
    Args:
        images_filepaths: list of image filepaths
        predicted_labels: Predicted Labels
    """
    rows = len(images_filepaths) // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i, image_filepath in enumerate(images_filepaths):
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        true_label = os.path.normpath(image_filepath).split(os.sep)[-2]
        predicted_label = predicted_labels[i] if predicted_labels else true_label
        color = "green" if true_label == predicted_label else "red"
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(predicted_label, color=color)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


def tensor_dataset(data: array, target:array):
    """
    Displays images in grid 
    Args:
        data: Data
        target: Target
    Returns: 
        tensordataset: Tensor Dataset
    """
    tensor_data = torch.Tensor(data) # transform to torch tensor
    tensor_target= torch.Tensor(target)
    tensordataset = TensorDataset(tensor_data,tensor_target)
    return tensordataset


class PILImageDataset(Dataset):

    def __init__(self, df:DataFrame, transform: transforms =None):
        self.df, self.paths, self.labels, self.transform = get_from_df_paths_targets( df, transform=transform)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = Image.open(self.paths[idx])
        label = torch.tensor(int(self.labels[idx]))
        if self.transform:
            image = self.transform(image)
        return image, label  
    

class CV2ImageDataset(Dataset):
    """
    OpenCV dataset to be used with albumentations
    """
    def __init__(self, df:DataFrame, transform: transforms =None, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.df, self.paths, self.labels, self.transform = get_from_df_paths_targets( df, transform=transform)
        self.device=device
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = torch.tensor(int(self.labels[idx]))
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label
    
class dataset_loader():
    """
    Data Loader object
    """
    def __init__(self, dataset: Dataset, batch_size:int = 1 , num_workers: int =1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
    def get_dataloader(self,):
        """
        Method to return dataloader
        Returns:
            self.loader: Dataset Loader
        """
        self.loader = DataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=True
                )
        return self.loader

    def check_dataloader_dimension(self):
        """
        Prints out the dimension of dataloader
        """
        for _, (data, target) in enumerate(self.loader):
            print('Data Shape of Dataloader is (data, target) : ', data.shape, target.shape)
            print('Data Type of Dataloader is (data, target) : ', type(data), type(target))
            torch.cuda.empty_cache()
            break