import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

pred_normal_file = ['./pred_normal_356/'+i for i in os.listdir('./pred_normal_356/') if '.jpg' in i]
pred_abnormal_file = ['./pred_abnormal_356/'+i for i in os.listdir('./pred_abnormal_356/') if '.jpg' in i]
pred_abnormal_mask_file = ['./pred_abnormal_mask_356/'+i for i in os.listdir('./pred_abnormal_mask_356/') if '.jpg' in i]

# Modify patterns
# Calculate area and setting threshold
# def calculate_area_and_filter(image, threshold):
#     contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     biggest_area_contour = ''; biggest_area = 0
#     for contour in contours:
#         if biggest_area < cv2.contourArea(contour):
#             biggest_area_contour = contour
#         if cv2.contourArea(contour) < threshold:
#             cv2.drawContours(image, contour, -1, (0, 0, 0), -1)
#         else:
#             cv2.drawContours(image, contour, 0, (255, 255, 255), 5)
#     # cv2.drawContours(image, biggest_area_contour, 0, (255, 255, 255), 5)
#     return image

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
    normal_dice = []
    for num in range(len(pred_normal_file)):
        normal_img = cv2.imread(pred_normal_file[num], cv2.IMREAD_GRAYSCALE)
        filename = pred_normal_file[num].split('pred_normal_356/')[1]
        if modify:
            # normal_img = calculate_area_and_filter(normal_img, area_threshold)
            normal_img = ero_and_dil(normal_img)
            plt.imsave('./pred_normal_356_modify/{}'.format(filename), normal_img, cmap="gray")
        normal_mask = np.zeros(normal_img.shape)
        normal_dice.append(dice_score(normal_img, normal_mask))

        # TN, FP
        if np.sum(normal_img)==0: TN += 1
        if np.sum(normal_img)!=0: FP += 1


    abnormal_dice = []
    for num in range(len(pred_abnormal_file)):
        abnormal_img = cv2.imread(pred_abnormal_file[num], cv2.IMREAD_GRAYSCALE)
        filename = pred_abnormal_file[num].split('pred_abnormal_356/')[1]
        if modify:
            # abnormal_img = calculate_area_and_filter(abnormal_img, area_threshold)
            abnormal_img = ero_and_dil(abnormal_img)
            plt.imsave('./pred_abnormal_356_modify/{}'.format(filename), abnormal_img, cmap="gray")
        abnormal_mask = cv2.imread(pred_abnormal_mask_file[num], cv2.IMREAD_GRAYSCALE)
        abnormal_dice.append(dice_score(abnormal_img, abnormal_mask))

        # TP, FN
        if np.sum(abnormal_img)!=0: TP += 1
        if np.sum(abnormal_img)==0: FN += 1

    # print('normal_dice: {}'.format(np.mean(normal_dice)))
    # print('abnormal_dice: {}'.format(np.mean(abnormal_dice)))
    # print('total_dice: {}'.format(np.mean(normal_dice+abnormal_dice)))
    # print('\n')
    normal_dice = round(np.mean(normal_dice), 3)
    abnormal_dice = round(np.mean(abnormal_dice), 3)
    total_dice = round(np.mean(normal_dice+abnormal_dice), 3)

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


