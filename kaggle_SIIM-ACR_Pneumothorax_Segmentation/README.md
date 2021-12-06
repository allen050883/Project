# SIIM-ACR Pneumothorax Segmentation  
Reference: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview  
  
## Environment and packages  
ubuntu 20.04 + cuda 11.1 + cudnn 8.0.5  

## Data preprocess   
train-valid split for 0.75, turn dcm files into jpg files  
  
## DataLoader  
Reading all files complete path, use cv2.imwrite open images  
Data Argumentation  
  
## Network  
ResUnet, using pytorch resnet-50 layer for pre-train model  
Reference: https://arxiv.org/pdf/1711.10684.pdf  
  
Optimizer using Ranger in torch-optimizer package  
Paper: New Deep Learning Optimizer, Ranger: Synergistic combination of RAdam + LookAhead for the best of both (2019)  
[https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d]  
Reference Code: https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer  
  
## Loss  
score: dice-score (from offical websites), but need to modify  
Combo loss: alpha * dice_loss + Beta * focal_loss + Gamma * binary_cross_entropy_loss  
Reference: https://arxiv.org/pdf/1805.02798.pdf  
  
  
# How to run the code?  
1. Install environment  


2. run preprocess  
```
python preprocess.py
```
generate 6 folder including:  
train_img_normal  
train_mask_normal  
train_img_abnormal  
train_mask_abnormal  
test_img  
test_mask  

3. Setting Config, and use pre-train weight or not  
```
python main.py
```
  
4. Result  
```
python predict.py
```
Predict the validation image, and draw in th folder.
