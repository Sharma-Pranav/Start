from torch.utils.data.dataloader import DataLoader
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from statistics import mode
from average_meter import AverageMeter
from model_class import NeuralNet
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class MetaLearn:
    def __init__(self, model_list:list=None, pre_transforms:list = None, tta_alb_transforms:list= None):
        super(MetaLearn, self).__init__()
        self.model_list = model_list
        self.pre_transforms = pre_transforms
        self.tta_transforms = tta_alb_transforms
        self.tta_accuracy_meter =  AverageMeter()
    def meta_learn(self):
        assert len(self.model_list) !=0 and (len(self.model_list)+1)% 2 ==0  

    def test_time_augmentation(self, model, test_df , tta_transforms,  device):
        self.tta_transforms= tta_transforms
        assert (len(self.tta_transforms))%2==0, 'Needs Even number of assertion for transforms with extra as is prediction'
        def get_output_for_transforms(copy_req_transforms,model,image,device):
                compose_aug = A.Compose(copy_req_transforms)
                transformed_image = compose_aug(image=image)["image"]
                transformed_image = torch.unsqueeze(transformed_image, 0)
                
                #print('tta device: ', device)
                transformed_image.to(device)
                model.to(device)
                
                #print(type(model), type(transformed_image))
                #print(model.fc1.weight.get_device())
                #print(transformed_image.get_device())
                model_output = model(transformed_image)
                prediction = torch.argmax(model_output, axis =1)
                return prediction

        test_df["tta_pred"] = ""
        for i in range(len(self.tta_transforms)+1):
            test_df['pred_aug_'+str(i)]= ""
        req_transforms = [A.Normalize(), ToTensorV2()]
        for index, row in test_df.iterrows():
            #print(row)
            image = cv2.imread(row['path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            list_predictions = []
            for i in range(len(self.tta_transforms)):
                copy_req_transforms = req_transforms.copy()
                copy_req_transforms.insert(0, self.tta_transforms[i])       
                pred = get_output_for_transforms(copy_req_transforms,model,image, device)
                #row['pred_aug_'+str(i)] = pred.cpu().numpy()
                test_df['pred_aug_'+str(i)].iloc[index] = int(pred.cpu().numpy())
                #print(pred.cpu())
                list_predictions.append(int(pred.cpu().numpy()))
            copy_req_transforms = req_transforms.copy()       
            pred = get_output_for_transforms(copy_req_transforms,model,image, device)
            #row['pred_aug_'+str(len(self.tta_transforms) + 1)] =pred.cpu()
            test_df['pred_aug_'+str(i)].iloc[index] = int(pred.cpu().numpy())
            list_predictions.append(int(pred.cpu().numpy()))
            #print('list_predictions : ', list_predictions)
            tta_pred = mode(list_predictions)
            #print('tta_pred : ', tta_pred)
            #row[["tta_pred"]] = tta_pred#int(tta_pred.cpu().numpy())
            test_df['tta_pred'].iloc[index] = tta_pred#.cpu().numpy()
            #if index==5:
            #    break
        #print(test_df["tta_pred"].to_list()[:5])
        #print(test_df['label'].to_list()[:5])
        list_bools = test_df["tta_pred"] == test_df['label']
        
        #print('list_bools : ', list_bools)
        #list_bools = test_df["tta_pred"][:5].to_list() == test_df['label'][:5].to_list()
        batch_correct = list_bools.sum()
        #print(batch_correct)
        #print(test_df.shape[0])
        self.tta_accuracy_meter.update(batch_correct, test_df.shape[0], 0)
        return  test_df, self.tta_accuracy_meter.return_current_avg()

    def generate_tta_dict_for_folds(self,val_df , tta_transforms, device):
            
        new_tta_dfs_dict = {}
        for i,  model in enumerate(self.model_list):
            model = self.model_list[i]

            nn_model = NeuralNet(model)
            model = nn_model.get_model()

            model = model.to(device)
            test_df, test_accuracy_tta = self.test_time_augmentation(model, val_df , tta_transforms, device)

            new_tta_dfs_dict[i] = [test_df, test_accuracy_tta]
            
        return new_tta_dfs_dict

    def meta_learn_on_dict(self, tta_dict):
        prep_dfs = []
        for i, key in enumerate(tta_dict.keys()):
            #print('Key :', key)
            df, acc = tta_dict[key]
            df = df.rename(columns=lambda x: x.replace('pred', str(i)+'_custom_'+str(i)))
            if i!=0:
                df = df.drop(['path', 'label'], axis=1)

            #print('df.shape : ', df.shape)
            prep_dfs.append(df)
        prep_dfs = pd.concat(prep_dfs, axis =1)
        #print('concatdf.shape : ', prep_dfs.shape)
        return prep_dfs
        