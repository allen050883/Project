import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn 
import torchvision.models as models
from model import Unet

from utils.dataloader import read_data_path, MaskDataset
from torch.utils.data import DataLoader
from utils.config import Config
from utils.loss import dice_score

# Hyperparameter
config = Config()
TRAIN_TEST_SPLIT = config.TRAIN_TEST_SPLIT
BATCH_SIZE_VALIDATION = config.BATCH_SIZE_VALIDATION
BATCH_SIZE_TESTING = config.BATCH_SIZE_TESTING
PRED_SAVE_DIR = config.PRED_SAVE_DIR
os.makedirs(PRED_SAVE_DIR, exist_ok=True)
INFERENCE_WEIGHT = config.INFERENCE_WEIGHT

# Use torch cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import Resnet-50 as base network, modify first layer
model_ft = models.resnet50(pretrained=True)
model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)

# Add Residual layer in unet
model = Unet(model_ft)
model.to(device)
if INFERENCE_WEIGHT:
    model.load_state_dict(torch.load(INFERENCE_WEIGHT))

# Read data path, make in dataloader
"""
read_data_path
    input: (float), the split of train and test
    return: (list, list, list), train & valid & test file path list
             list -> (img_path, mask_path)
"""
training_list, validation_list, testing_list = read_data_path(TRAIN_TEST_SPLIT)

val_dataset = MaskDataset(validation_list)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_VALIDATION, shuffle=False, drop_last=True)


# Confusion matrix, postive = abnormal; negative = normal
TP = 0; FP = 0
FN = 0; TN = 0
"""
TP => mask: abnormal, pred: abnormal
FP => mask: normal, pred: abnormal
FN => mask: abnormal, pred: normal
TN => mask: normal, pred: normal 
"""

dice_score_list = []
number = 0
with torch.no_grad():
    for imgs, masks in val_loader:
        imgs_gpu = imgs.to(device)
        outputs = model(imgs_gpu)
        outputs = torch.round(outputs) * 255
        masks = masks.to(device)

        # Dice score list
        dice_scores = dice_score(outputs, masks)
        dice_score_list.extend([dice_scores.item()])

        for index in range(BATCH_SIZE_VALIDATION):
            img_origin = np.reshape(imgs_gpu[index].cpu().numpy(), (256, 256))
            pred_img = np.reshape(outputs[index].cpu().numpy(), (256, 256))
            mask_img = np.reshape(masks[index].cpu().numpy()*255, (256, 256))

            # Confusion Matrix
            if np.sum(mask_img)!=0 and np.sum(pred_img)!=0: TP += 1
            if np.sum(mask_img)==0 and np.sum(pred_img)!=0: FP += 1
            if np.sum(mask_img)!=0 and np.sum(pred_img)==0: FN += 1
            if np.sum(mask_img)==0 and np.sum(pred_img)==0: TN += 1

            number += 1
            print(number)

            if np.all(mask_img==0):
                plt.imsave('./pred_normal_356' + '/' + str(number) + '.jpg', pred_img, cmap="gray") 
            else:
                plt.imsave('./pred_abnormal_356' + '/' + str(number) + '.jpg', pred_img, cmap="gray")
                plt.imsave('./pred_abnormal_mask_356' + '/' + str(number) + '.jpg', mask_img, cmap="gray")
            plt.close()

print('TP: {}, FP: {}, FN: {}, TN: {}'.format(TP, FP, FN, TN))
print('Accuracy: {}'.format((TP+TN)/(TP+FP+FN+TN)))
print('Precision: {}'.format((TP)/(TP+FP)))
print('Recall: {}'.format((TP)/(TP+FN)))  # Medical domain
print('\n')
print('Sensitivity: {}'.format((TP)/(TP+FN)))
print('Specificity: {}'.format((TN)/(FP+TN)))
print('Dice score: {}'.format(np.mean(np.array(dice_score_list))))
