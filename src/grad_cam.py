# Based on https://github.com/yaleCat/Grad-CAM-pytorch
import cv2
import numpy as np
import torch

img_max_val = 255 
img_size = (512, 512)
class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model):
        self.model = model
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        features = []
        self.gradients = []
        for idx, module in self.model._modules.items():
            if idx == 'conv_list':
                for i, elem  in enumerate(module):
                    x = self.model.feed_forward(x, elem, conv =True)
                    if i==len(module)-1:
                        x.register_hook(self.save_gradient)
                        features += [x] 
            elif idx =='fc_list':
                for i, elem  in enumerate(module):
                    if i ==0:
                        x = torch.flatten(x, 1) 
                    x = self.model.feed_forward(x, elem, conv =False)
        return features, x

def show_cam_on_image(img, mask, name):
    heatmap = cv2.applyColorMap(np.uint8(img_max_val * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / img_max_val
    img = np.float32(img) / img_max_val

    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return  np.uint8(img_max_val * cam), name, heatmap*img_max_val

class GradCam:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.extractor = FeatureExtractor(self.model)

    def __call__(self, inputs, index=None):
        mask_dict = {}
        if self.cuda:
            features, output = self.extractor(inputs.cuda())
        else:
            features, output = self.extractor(inputs)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward()
        
        self.model.zero_grad()

        for idx, feature in enumerate(features):
            grads_val = self.extractor.gradients[len(features)-1-idx].cpu().data.numpy()

            target = feature
            target = target.cpu().data.numpy()[0, :]
            weights = np.mean(grads_val, axis=(2, 3))[0, :]
            cam = np.zeros(target.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * target[i, :, :]
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, img_size)
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
            mask_dict['feature_idx_'+str(idx)] = cam
        return mask_dict
