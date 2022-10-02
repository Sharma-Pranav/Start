#from torch.utils.data.dataloader import DataLoader
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from statistics import mode
from average_meter import AverageMeter
from model_class import NeuralNet
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
#import matplotlib.pyplot as plt
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class MetaLearn:
    def __init__(self, model_list:list=None, pre_transforms:list = None, tta_alb_transforms:list= None):
        """Initialises metalearning object
        Args:
            model_list ([list]): [list of models for metalearning]
            pre_transforms ([list]): [list of transforms to be always applied]
            tta_alb_transforms ([list]): [list of test time augmentations to be applied]
        """
        super(MetaLearn, self).__init__()
        self.model_list = model_list
        self.pre_transforms = pre_transforms
        self.tta_transforms = tta_alb_transforms
        self.tta_accuracy_meter =  AverageMeter()
    def meta_learn(self):
        """
        Check for ideal MetaLearning of objects
        """
        assert len(self.model_list) !=0 and (len(self.model_list)+1)% 2 ==0, 'Metalearning is better on odd number of models'

    def test_time_augmentation(self, model, test_df , tta_transforms,  device):
        """Translates the state (from observation environment returns Node and scale inputs)
        Args:
            model: Deep Learning Model 
            test_df: dataframe for making test time dataframe
            tta_transforms : List of test time transforms
            device : Device on which the test time augmentation is to be done 
        Returns:
            test_df: Appended dataframe with test time augmentation 
            current_accuracy_average : Current Accuracy with test time augmentation
        """
        self.tta_transforms= tta_transforms
        assert (len(self.tta_transforms))%2==0, 'Needs Even number of assertion for transforms with extra as is prediction'
        def get_output_for_transforms(copy_req_transforms,model,image,device):
                compose_aug = A.Compose(copy_req_transforms)
                transformed_image = compose_aug(image=image)["image"]
                transformed_image = torch.unsqueeze(transformed_image, 0)
                transformed_image = transformed_image.to(device)
                model_output = model(transformed_image)
                prediction = torch.argmax(model_output, axis =1)
                
                return prediction

        test_df["tta_pred"] = ""
        for i in range(len(self.tta_transforms)):
            test_df['pred_aug_'+str(i)]= ""
        req_transforms = [A.Normalize(), ToTensorV2()]
        for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
            image = cv2.imread(row['path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            list_predictions = []
            for i in range(len(self.tta_transforms)):
                copy_req_transforms = req_transforms.copy()
                copy_req_transforms.insert(0, self.tta_transforms[i])  
                pred = get_output_for_transforms(copy_req_transforms,model.to(device), image, device)
                test_df['pred_aug_'+str(i)].iloc[index] = int(pred.cpu().numpy())
                list_predictions.append(int(pred.cpu().numpy()))
            copy_req_transforms = req_transforms.copy()       
            pred = get_output_for_transforms(copy_req_transforms,model,image, device)
            test_df['pred_aug_'+str(i)].iloc[index] = int(pred.cpu().numpy())
            list_predictions.append(int(pred.cpu().numpy()))
            tta_pred = mode(list_predictions)
            test_df['tta_pred'].iloc[index] = tta_pred
        list_bools = test_df["tta_pred"] == test_df['label']
        batch_correct = list_bools.sum()
        self.tta_accuracy_meter.update(batch_correct, test_df.shape[0], 0)
        current_accuracy_average = self.tta_accuracy_meter.return_current_avg()
        return  test_df, current_accuracy_average #self.tta_accuracy_meter.return_current_avg()

    def generate_tta_dict_for_folds(self,val_df , tta_transforms, device):
        """Generates Test Time Augmentation dictionary containining test time augmented dataframes corresponding to all models in the model list
        Args:
            val_df: Validation Datatframe
            tta_transforms : List of test time transforms
            device : Device on which the test time augmentation is to be done 
        Returns:
            new_tta_dfs_dict: Generated dictionary of test time augmentations
        """    
        new_tta_dfs_dict = {}
        for i,  model in enumerate(self.model_list):
            model = self.model_list[i]

            nn_model = NeuralNet(model)
            model = nn_model.get_model()

            model = model.to(device)
            test_df, test_accuracy_tta = self.test_time_augmentation(model, val_df , tta_transforms, device)

            new_tta_dfs_dict[i] = [test_df, test_accuracy_tta]
            model.cpu()
            
        return new_tta_dfs_dict

    def create_meta_learn_labels_on_dict(self, tta_dict):
        """Prepare metalearning dataframe from the given Test Time Augmentation Dictionary
        Args:
            tta_dict: Test time Augmentation Dictionary
        Returns:
            prep_dfs: Prepared Dataframe for Metalearning
        """    
        prep_dfs = []
        for i, key in tqdm(enumerate(tta_dict.keys())):
            df, acc = tta_dict[key]
            columns=lambda x: x.replace('pred', str(i)+'_custom_'+str(i))
            df = df.rename(columns = columns)
            if i!=0:
                df = df.drop(['path', 'label'], axis=1)
            prep_dfs.append(df)
        prep_dfs = pd.concat(prep_dfs, axis =1)
        return prep_dfs
        
    def get_data_from_tta_dict(self, tta_df, query_string):
        """Prepare metalearning dataframe from the given Test Time Augmentation Dictionary
        Args:
            tta_dict: Test time Augmentation Dictionary
            query_string : Query String contained in predictions
        Returns:
            data: Training Data
            labels: Training Labels
        """   
        column_name_list = tta_df.columns.tolist()
        columns = [name for name in column_name_list if query_string in name ]
        label_column = ['label']
        df_data = tta_df[columns]
        df_label = tta_df[label_column]
        data = df_data.to_numpy()
        labels = df_label.to_numpy()
        return data, labels

    def do_meta_learning_on_tta_dicts(self, classifier, train_tta_df, val_tta_df, query_string):
        """Do Metalearning on metalearning data
        Args:
            classifier: Scikit Learn
            train_tta_df : Training Metalearning Dataframe 
            val_tta_df : Validation Metalearning Dataframe
            query_string: Query String contained in predictions
        Returns:
            val_acc: Metalearning Validation Accuracy
            val_loss: Validation loss 
        """  

        train_data, train_labels = self.get_data_from_tta_dict(train_tta_df, query_string)
        val_data, val_labels = self.get_data_from_tta_dict(val_tta_df, query_string)
        classifier.fit(train_data, train_labels)
        val_predictions = classifier.predict(val_data)
        val_acc = accuracy_score(val_labels, val_predictions)
        val_prob_predictions = classifier.predict_proba(val_data)
        val_loss = log_loss(val_labels, val_prob_predictions)

        return val_acc, val_loss