import os
import random

# Train and Test Img and Mask directory
TRAIN_IMG_NORMAL_DIR = './train_img_normal/'
TRAIN_MASK_NORMAL_DIR = './train_mask_normal/'
TRAIN_IMG_ABNORMAL_DIR = './train_img_abnormal/'
TRAIN_MASK_ABNORMAL_DIR = './train_mask_abnormal/'
TEST_IMG_DIR = './test_img/'
TEST_MASK_DIR = './test_mask/'



def read_data_path(TRAIN_TEST_SPLIT=0.75):
    # Read data list and sort
    train_img_normal_dir = [os.path.join(TRAIN_IMG_NORMAL_DIR, i) for i in os.listdir(TRAIN_IMG_NORMAL_DIR)]
    train_img_normal_dir.sort()
    train_mask_normal_dir = [os.path.join(TRAIN_MASK_NORMAL_DIR, i)for i in os.listdir(TRAIN_MASK_NORMAL_DIR)]
    train_mask_normal_dir.sort()
    train_img_abnormal_dir = [os.path.join(TRAIN_IMG_ABNORMAL_DIR, i) for i in os.listdir(TRAIN_IMG_ABNORMAL_DIR)]
    train_img_abnormal_dir.sort()
    train_mask_abnormal_dir = [os.path.join(TRAIN_MASK_ABNORMAL_DIR, i) for i in os.listdir(TRAIN_MASK_ABNORMAL_DIR)]
    train_mask_abnormal_dir.sort()

    # Tuple for img and mask
    train_normal_list = [(data, label) for data, label in zip(train_img_normal_dir, train_mask_normal_dir)]
    train_abnormal_list = [(data, label) for data, label in zip(train_img_abnormal_dir, train_mask_abnormal_dir)]
    # Random list
    random.seed(2021)
    random.shuffle(train_normal_list)
    random.seed(2021)
    random.shuffle(train_abnormal_list)
    # Split train valid data
    training_list = train_normal_list[:int(len(train_normal_list)*TRAIN_TEST_SPLIT)] + train_abnormal_list[:int(len(train_abnormal_list)*TRAIN_TEST_SPLIT)]
    validation_list = train_normal_list[int(len(train_normal_list)*TRAIN_TEST_SPLIT):] + train_abnormal_list[int(len(train_abnormal_list)*TRAIN_TEST_SPLIT):]


    # Test data
    testing_list = [os.path.join(TEST_IMG_DIR, i) for i in os.listdir(TEST_IMG_DIR)]
    testing_list.sort()

    return training_list, validation_list, testing_list




import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MaskDataset(Dataset):
    def __init__(self, data_list, transforms=None):
        self.data_list = data_list
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.data_list[index][0]
        mask_path = self.data_list[index][1]

        # Load image data
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize image
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

        # Data Argumentation
        if self.transforms:
            sample = {
                "image": img,
                "mask": mask
            }
            sample = self.transforms(**sample)
            img = sample["image"]
            mask = sample["mask"]

        # Transger to tensor
        img = np.expand_dims(img, axis=-1)/255.0
        img = self.toTensor(img).float()

        mask = np.expand_dims(mask, axis=-1)/255.0
        mask = self.toTensor(mask).float()
        
        return img, mask
            
    def __len__(self):
        return len(self.data_list)

    def toTensor(self, np_array, axis=(2,0,1)):
        return torch.tensor(np_array).permute(axis)

