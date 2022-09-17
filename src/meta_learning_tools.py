from torch.utils.data.dataloader import DataLoader
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from statistics import mode
from average_meter import AverageMeter
class MetaLearn:
    def __init__(self, model_list:list=None, pre_transforms:list = None, tta_alb_transforms:list= None):
        super(MetaLearn, self).__init__()
        self.model_list = model_list
        self.pre_transforms = pre_transforms
        self.tta_transforms = tta_alb_transforms
        self.tta_accuracy_meter =  AverageMeter()
    def meta_learn(self):
        assert len(self.model_list) !=0 and (len(self.model_list)+1)% 2 ==0  

    def test_time_augmentation(self, model, test_df , device):
        assert (len(self.tta_transforms))%2==0, 'Needs Even number of assertion for transforms with extra as is prediction'
        def get_output_for_transforms(copy_req_transforms,model,image):
                compose_aug = A.Compose(copy_req_transforms)
                transformed_image = compose_aug(image)["image"]
                model_output = model(transformed_image.to(device))
                prediction = torch.argmax(model_output, axis =1)
                return prediction

        test_df["tta_pred"] = ""
        for i in range(len(self.tta_transforms)+1):
            test_df['pred_aug_'+str(i)]= ""
        req_transforms = [A.Normalize(), ToTensorV2()]
        for index, row in test_df.iterrows():
            image = cv2.imread(path = row['path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            list_predictions = []
            for i in range(len(self.tta_transforms)):
                copy_req_transforms = req_transforms.copy()
                copy_req_transforms.insert(0, self.tta_transforms[i])       
                pred = get_output_for_transforms(copy_req_transforms,model,image)
                row['pred_aug_'+str(i)] =pred
                list_predictions.append(pred)
            copy_req_transforms = req_transforms.copy()       
            pred = get_output_for_transforms(copy_req_transforms,model,image)
            row['pred_aug_'+str(len(self.tta_transforms) + 1)] =pred
            list_predictions.append(pred)
            tta_pred = mode(list_predictions)
            row[["tta_pred"]] = tta_pred
        batch_correct = (test_df["tta_pred"].to_list() == test_df['label'].to_list()).sum()
        self.tta_accuracy_meter.update(batch_correct, test_df.shape[0], 0)
        return  test_df