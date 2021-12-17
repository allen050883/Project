import os
import torch
import torch.nn as nn 
import torchvision.models as models
from torch.utils.data import DataLoader
import torch_optimizer as optim

from model import Unet
from torchsummary import summary

from utils.plot_network import Plot_model
from utils.dataloader import read_data_path, MaskDataset
from utils.loss import dice_score, get_dice_loss, get_focal_loss, combo_loss
import albumentations as A


import csv
from tqdm import tqdm
import numpy as np
from utils.config import Config

config = Config()

# Directory or Path Name
MODEL_SAVE_DIR = config.MODEL_SAVE_DIR
PRETRAIN_WEIGHT = config.PRETRAIN_WEIGHT
LEARNING_HISTORY_FILENAME = config.LEARNING_HISTORY_FILENAME
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Hyperparameter
TRAIN_TEST_SPLIT = config.TRAIN_TEST_SPLIT
UPSAMPLEING = config.UPSAMPLEING
START_EPOCH = config.START_EPOCH
END_EPOCH = config.END_EPOCH
LERANING_RATE = config.LERANING_RATE
BATCH_SIZE_TRAINING = config.BATCH_SIZE_TRAINING
BATCH_SIZE_VALIDATION = config.BATCH_SIZE_VALIDATION
PRETRAIN_WEIGHT = config.PRETRAIN_WEIGHT

# Use torch cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import Resnet-50 as base network, modify first layer
model_ft = models.resnet50(pretrained=True)
model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)

# Add Residual layer in unet, use pre-train weight or not
model = Unet(model_ft)
model.to(device)
if PRETRAIN_WEIGHT:
    model.load_state_dict(torch.load(PRETRAIN_WEIGHT))

# Use inital weight, leaky_relu use he_normalize
def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight.data)
model.apply(initialize_weights)

# Summary and plot model
print(summary(model,input_size=(1, 256, 256)))
# Plot_model('ResUnet network', model)



# Read data path, make in dataloader
"""
read_data_path
    input: (float), the split of train and test
    return: (list, list, list), train & valid & test file path list
             list -> (img_path, mask_path)
"""
training_list, validation_list, testing_list = read_data_path(TRAIN_TEST_SPLIT, UPSAMPLEING)
print('Training list count: {}'.format(len(training_list)))
print('Validation list count: {}'.format(len(validation_list)))
print('Testing list count: {}'.format(len(testing_list)))

# Data Argumentation
train_transform = A.Compose([
    A.HorizontalFlip(),
    # A.OneOf([
    #     A.RandomContrast(),
    #     A.RandomGamma(),
    #     A.RandomBrightness(),
    #     ], p=0.5),
    A.ShiftScaleRotate(),
])

train_dataset = MaskDataset(training_list, train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAINING, shuffle=True, drop_last=True)

val_dataset = MaskDataset(validation_list)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_VALIDATION, shuffle=True, drop_last=True)


# Model setting
train_params = [param for param in model.parameters() if param.requires_grad]

# Optimizer
# optimizer = torch.optim.Adam(train_params, lr=LERANING_RATE, betas=(0.9, 0.99))
optimizer = optim.Ranger(
    train_params,
    lr=LERANING_RATE,
    alpha=0.5,
    k=6,
    N_sma_threshhold=5,
    betas=(.95, 0.999),
    eps=1e-5,
    weight_decay=0
)
# Pytorch model mode
model.train()


# Record learning history in csv
with open(LEARNING_HISTORY_FILENAME, mode='w') as csv_file:
    fieldnames = ['epoch', 'train_dice_score', 'val_dice_score']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Start to training from START_EPOCH, END_EPOCH
    for epoch in range(START_EPOCH, END_EPOCH):
        number_iter = 1

        train_dice_score = []
        train_losses = []
        # Use tqdm as learning programming
        with tqdm(train_loader, unit="batch") as tepoch:
            for imgs, masks in tepoch:
                tepoch.set_description(f"Epoch {epoch}, interation {number_iter}")
                
                optimizer.zero_grad()
                imgs_gpu = imgs.to(device)
                outputs = model(imgs_gpu)
                masks = masks.to(device)

                # Calculate dicr score and loss function
                dice_scores = dice_score(outputs, masks)
                loss = combo_loss(outputs, masks)

                train_dice_score.extend([dice_scores.item()])
                train_losses.extend([loss.item()])
                
                # Back propagation
                loss.backward()
                optimizer.step()
                # nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                tepoch.set_postfix(loss=loss.item(), dice_score=dice_scores.item())
                number_iter += 1
                # break

            train_dice_score = np.mean(np.array(train_dice_score))
            train_losses = np.mean(np.array(train_losses))
            

        # Validation inference
        val_dice_score = []
        val_losses = []
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader):
                imgs_gpu = imgs.to(device)
                outputs = model(imgs_gpu) 
                masks = masks.to(device)
                
                dice_scores = dice_score(outputs, masks)
                loss = combo_loss(outputs, masks)
                
                val_dice_score.extend([dice_scores.item()])
                val_losses.extend([loss.item()])
                # break

            val_dice_score = np.mean(np.array(val_dice_score))
            val_losses = np.mean(np.array(val_losses))
            
            # Show training and validation result, and save model
            print(f"Epoch {epoch}, Train loss {train_losses:0.4f}, train dice score: {train_dice_score:0.4f}, Validation loss {val_losses:0.4f}, validation dice score: {val_dice_score:0.4f}")
            path = "./{}/model_epoch_{:.1f}_train_dice_score_{:0.4f}_val_dice_score_{:0.4f}.pth".format(MODEL_SAVE_DIR, epoch, train_dice_score, val_dice_score)
            torch.save(model.state_dict(), path)
        

            # Write result in csv file
            writer.writerow({
                "epoch": epoch,
                "train_dice_score": train_dice_score,
                "val_dice_score": val_dice_score
            })
