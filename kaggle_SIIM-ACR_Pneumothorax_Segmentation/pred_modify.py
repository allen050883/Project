import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

pred_normal_file = ['./pred_normal_301/'+i for i in os.listdir('./pred_normal_301/') if '.jpg' in i]
pred_abnormal_file = ['./pred_abnormal_301/'+i for i in os.listdir('./pred_abnormal_301/') if '.jpg' in i]
pred_abnormal_mask_file = ['./pred_abnormal_mask_301/'+i for i in os.listdir('./pred_abnormal_mask_301/') if '.jpg' in i]


# Erosion and Dilation
def ero_and_dil(image):
    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(image, kernel, iterations = 1)

    kernel = np.ones((7,7), np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations = 1)

    return dilation


def dice_score(inputs, targets, smooth=1):
    #flatten label and prediction tensors
    inputs = inputs.flatten()
    targets = targets.flatten()

    intersection = (inputs * targets).sum()                            
    dice_score = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
    
    return dice_score

def cal_score(pred_normal_file, pred_abnormal_file, pred_abnormal_mask_file, area_threshold, modify=True):
    # Confusion matrix, postive = abnormal; negative = normal
    TP = 0; FP = 0
    FN = 0; TN = 0

    # modify prediction or not
    normal_dice_list = []
    for num in range(len(pred_normal_file)):
        normal_img = cv2.imread(pred_normal_file[num], cv2.IMREAD_GRAYSCALE)
        filename = pred_normal_file[num].split('pred_normal_301/')[1]
        if modify:
            # normal_img = calculate_area_and_filter(normal_img, area_threshold)
            normal_img = ero_and_dil(normal_img)
            plt.imsave('./pred_normal_301_modify/{}'.format(filename), normal_img, cmap="gray")
        normal_mask = np.zeros(normal_img.shape)
        normal_dice_list.append(dice_score(normal_img, normal_mask))

        # TN, FP
        if np.sum(normal_img)==0: TN += 1
        if np.sum(normal_img)!=0: FP += 1


    abnormal_dice_list = []
    for num in range(len(pred_abnormal_file)):
        abnormal_img = cv2.imread(pred_abnormal_file[num], cv2.IMREAD_GRAYSCALE)
        filename = pred_abnormal_file[num].split('pred_abnormal_301/')[1]
        if modify:
            # abnormal_img = calculate_area_and_filter(abnormal_img, area_threshold)
            abnormal_img = ero_and_dil(abnormal_img)
            plt.imsave('./pred_abnormal_301_modify/{}'.format(filename), abnormal_img, cmap="gray")
        abnormal_mask = cv2.imread(pred_abnormal_mask_file[num], cv2.IMREAD_GRAYSCALE)
        abnormal_dice_list.append(dice_score(abnormal_img, abnormal_mask))

        # TP, FN
        if np.sum(abnormal_img)!=0: TP += 1
        if np.sum(abnormal_img)==0: FN += 1

    # print('normal_dice: {}'.format(np.mean(normal_dice)))
    # print('abnormal_dice: {}'.format(np.mean(abnormal_dice)))
    # print('total_dice: {}'.format(np.mean(normal_dice+abnormal_dice)))
    # print('\n')
    # print(normal_dice)
    normal_dice = round(np.mean(normal_dice_list), 3)
    abnormal_dice = round(np.mean(abnormal_dice_list), 3)
    total_dice = round(np.mean(normal_dice_list+abnormal_dice_list), 3)

    # print('TP: {}, FP: {}, FN: {}, TN: {}'.format(TP, FP, FN, TN))
    # print('Accuracy: {}'.format((TP+TN)/(TP+FP+FN+TN)))
    # print('Precision: {}'.format((TP)/(TP+FP)))
    # print('Recall: {}'.format((TP)/(TP+FN)))  # Medical domain
    # print('\n')
    # print('Sensitivity: {}'.format((TP)/(TP+FN)))
    # print('Specificity: {}'.format((TN)/(FP+TN)))
    Accuracy = round((TP+TN)/(TP+FP+FN+TN), 3)
    Precision = round((TP)/(TP+FP), 3)
    Recall = round((TP)/(TP+FN), 3)
    Specificity = round((TN)/(FP+TN), 3)

    return (normal_dice, abnormal_dice, total_dice, Accuracy, Precision, Recall, Specificity)

score = {}
(normal_dice, abnormal_dice, total_dice, Accuracy, Precision, Recall, Specificity) = cal_score(pred_normal_file, pred_abnormal_file, pred_abnormal_mask_file, 0, False)
score['origin'] = (normal_dice, abnormal_dice, total_dice, Accuracy, Precision, Recall, Specificity)

for i in range(1, 2):
    (normal_dice, abnormal_dice, total_dice, Accuracy, Precision, Recall, Specificity) = cal_score(pred_normal_file, pred_abnormal_file, pred_abnormal_mask_file, i, True)
    score[str(i)] = (normal_dice, abnormal_dice, total_dice, Accuracy, Precision, Recall, Specificity)

for key, value in score.items():
    print(key, value)


