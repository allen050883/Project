import os
import pandas as pd
import numpy as np
import cv2
import pydicom
from utils.utils import rle2mask


# Train and Test dcm file directory
TRAIN_DCM_DIR = './dicom-images-train/'
TEST_DCM_DIR = './dicom-images-test/'

# Train and Test Img and Mask directory
TRAIN_IMG_NORMAL_DIR = './train_img_normal/'
TRAIN_MASK_NORMAL_DIR = './train_mask_normal/'
TRAIN_IMG_ABNORMAL_DIR = './train_img_abnormal/'
TRAIN_MASK_ABNORMAL_DIR = './train_mask_abnormal/'
TEST_IMG_DIR = './test_img/'
TEST_MASK_DIR = './test_mask/'


def Plot_Img_and_Mask(filepath, img_dict, img_dir, mask_dir):
    """
    input:
        filepath: dict, file name: file complete path
        img_dict: dict, Image ID(file name): mask location(list)
        img_dir: string, save img directory
        mask_dir: string, save mask directory
    """
    count = 0
    for filename, file_path in filepath.items():
        if img_dict:
            # Plot img
            if filename in img_dict:
                ds = pydicom.read_file(file_path)
                img = ds.pixel_array
                img_shape = img.shape
                cv2.imwrite(img_dir + filename + '.jpg', img*255)

                count += 1
                if count % 100 == 0: print("Finish draw {}".format(count))

            # Plot mask
            if mask_dir != None:
                if filename in img_dict and img_dict != None:
                    if img_dict[filename][0] == '-1':
                        # '-1' means healthy, output black mask
                        mask_img = np.zeros(img_shape)
                        cv2.imwrite(mask_dir + filename + '_mask.jpg', mask_img*255)

                    else:
                        # For loop if many mask, then sum them
                        sum_img = np.zeros(img_shape)
                        for rle in img_dict[filename]:
                            mask_img = rle2mask(rle, img_shape)
                            sum_img += mask_img
                        cv2.imwrite(mask_dir + filename + '_mask.jpg', sum_img*255)

        else:
            ds = pydicom.read_file(file_path)
            img = ds.pixel_array
            img_shape = img.shape
            cv2.imwrite(img_dir + filename + '.jpg', img*255)
            
            count += 1
            if count % 100 == 0: print("Finish draw {}".format(count))
        


def TrainFileInfo(plot_img):
    # Read train-rle.csv sepearate normal and abnormal list
    train_rle_csv = pd.read_csv('train-rle.csv', skipinitialspace=True)
    ImageId = list(train_rle_csv['ImageId']) # 12954
    EncodedPixels = list(train_rle_csv['EncodedPixels'])

    # Inital dict -> ImageId[num]:[]
    train_normal_dict = {ImageId[num]:[] for num in range(len(ImageId)) if EncodedPixels[num]=='-1'}
    train_abnormal_dict = {ImageId[num]:[] for num in range(len(ImageId)) if EncodedPixels[num]!='-1'}

    # Add mask in dict, maybe two mask in one picture
    for num in range(len(ImageId)):
        if EncodedPixels[num]!='-1':
            train_abnormal_dict[ImageId[num]].append(EncodedPixels[num])
        else:
            train_normal_dict[ImageId[num]].append(EncodedPixels[num])
        if num%100==0: print(num)

    # print(train_abnormal_dict); exit()
    print('Train normal: {}'.format(len(train_normal_dict)))     # 9378
    print('Train abnormal: {}'.format(len(train_abnormal_dict))) # 2669

    # Read all dcm file path
    filepath = {}
    for root, dirs, files in os.walk(TRAIN_DCM_DIR):
        for f in files:
            filepath[f.split('.dcm')[0]] = os.path.join(root, f)

    # Plot img and mask
    if plot_img == True:
        Plot_Img_and_Mask(filepath, train_normal_dict, TRAIN_IMG_NORMAL_DIR, TRAIN_MASK_NORMAL_DIR)
        print('Plot train normal finished!')
        Plot_Img_and_Mask(filepath, train_abnormal_dict, TRAIN_IMG_ABNORMAL_DIR, TRAIN_MASK_ABNORMAL_DIR)
        print('Plot train abnormal finished!')

    return train_normal_dict, train_abnormal_dict


def Test2Img(plot_img):
    filepath = {}
    for root, dirs, files in os.walk(TEST_DCM_DIR):
        for f in files:
            filepath[f.split('.dcm')[0]] = os.path.join(root, f)

    # Plot test folder img
    if plot_img == True:
        Plot_Img_and_Mask(filepath, None, TEST_IMG_DIR, None)
        print('Plot test finished!')


if __name__ == '__main__':
    # Create img and mask foler if not exist
    if not os.path.exists(TRAIN_IMG_NORMAL_DIR): os.mkdir(TRAIN_IMG_NORMAL_DIR)
    if not os.path.exists(TRAIN_MASK_NORMAL_DIR): os.mkdir(TRAIN_MASK_NORMAL_DIR)
    if not os.path.exists(TRAIN_IMG_ABNORMAL_DIR): os.mkdir(TRAIN_IMG_ABNORMAL_DIR)
    if not os.path.exists(TRAIN_MASK_ABNORMAL_DIR): os.mkdir(TRAIN_MASK_ABNORMAL_DIR)
    if not os.path.exists(TEST_IMG_DIR): os.mkdir(TEST_IMG_DIR)
    if not os.path.exists(TEST_MASK_DIR): os.mkdir(TEST_MASK_DIR)

    # Read train dcm files and unify
    TrainFileInfo(plot_img=False)

    # Transfer test dcm in test_img folder
    Test2Img(plot_img=False) #3205


