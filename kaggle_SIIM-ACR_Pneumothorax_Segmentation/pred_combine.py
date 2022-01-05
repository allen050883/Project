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


# Erosion and Dilation
def ero_and_dil(image):
    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(image.copy(), kernel, iterations = 1)

    kernel = np.ones((7,7), np.uint8)
    dilation = cv2.dilate(erosion.copy(), kernel, iterations = 1)

    return dilation

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

            number += 1
            print(number)

            if np.all(mask_img==0):
                fig, (ax1, ax2, ax3) = plt.subplots( nrows=1, ncols=3 )
                ax1.imshow(img_origin, cmap=plt.cm.bone)

                ax2.imshow(img_origin, cmap=plt.cm.bone)
                ax2.grid(False)
                mask_img = ((mask_img / 255.) - 1) * (-1)
                mask_img = np.concatenate([mask_img.reshape(256, 256, 1), np.ones([256, 256, 1]), mask_img.reshape(256, 256, 1)], axis=-1)
                mask_img = np.array(mask_img, np.float32)
                ax2.imshow(mask_img, alpha = 0.3)
                

                ax3.imshow(img_origin, cmap=plt.cm.bone)
                ax3.grid(False)

                ax3.imshow(mask_img, alpha = 0.3)
                pred_img_3 = ero_and_dil(pred_img)
                pred_img_3 = ((pred_img_3 / 255.) - 1) * (-1)
                pred_img_3 = np.concatenate([np.ones([256, 256, 1]),pred_img_3.reshape(256, 256, 1), np.ones([256, 256, 1])], axis=-1)
                pred_img_3 = np.array(pred_img_3, np.float32)
                ax3.imshow(pred_img_3, alpha = 0.3)

                fig.savefig('{}/{}.jpg'.format('./pred_normal_301_combine_1', number))
                plt.close(fig)

            else:
                fig, (ax1, ax2, ax3) = plt.subplots( nrows=1, ncols=3 )
                ax1.imshow(img_origin, cmap=plt.cm.bone)

                ax2.imshow(img_origin, cmap=plt.cm.bone)
                ax2.grid(False)
                mask_img = ((mask_img / 255.) - 1) * (-1)
                mask_img = np.concatenate([mask_img.reshape(256, 256, 1), np.ones([256, 256, 1]), mask_img.reshape(256, 256, 1)], axis=-1)
                mask_img = np.array(mask_img, np.float32)
                ax2.imshow(mask_img, alpha = 0.3)
                

                ax3.imshow(img_origin, cmap=plt.cm.bone)
                ax3.grid(False)

                ax3.imshow(mask_img, alpha = 0.3)
                pred_img_3 = ero_and_dil(pred_img)
                pred_img_3 = ((pred_img_3 / 255.) - 1) * (-1)
                pred_img_3 = np.concatenate([np.ones([256, 256, 1]),pred_img_3.reshape(256, 256, 1), np.ones([256, 256, 1])], axis=-1)
                pred_img_3 = np.array(pred_img_3, np.float32)
                ax3.imshow(pred_img_3, alpha = 0.3)

                fig.savefig('{}/{}.jpg'.format('./pred_abnormal_301_combine_1', number))
                plt.close(fig)
