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

# Hyperparameter
config = Config()
TRAIN_TEST_SPLIT = config.TRAIN_TEST_SPLIT
BATCH_SIZE_VALIDATION = config.BATCH_SIZE_VALIDATION
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


# Visualize predict img
number = 0
with torch.no_grad():
    for imgs, masks in val_loader:
        imgs_gpu = imgs.to(device)
        outputs = model(imgs_gpu)
        outputs = torch.round(outputs) * 255
        masks = masks.to(device)

        for index in range(BATCH_SIZE_VALIDATION):
            img_origin = np.reshape(imgs_gpu[index].cpu().numpy(), (512, 512))
            pred_img = np.reshape(outputs[index].cpu().numpy(), (512, 512))
            mask_img = np.reshape(masks[index].cpu().numpy()*255, (512, 512))

            if np.all(mask_img==0):
                continue
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(img_origin, cmap=plt.cm.bone)
            ax2.imshow(mask_img, cmap="gray")
            ax3.imshow(pred_img, cmap="gray")
            number += 1
            print(number)

            fig.savefig(PRED_SAVE_DIR + '/' + str(number) + '.jpg', dpi=200) 
            plt.close(fig)

