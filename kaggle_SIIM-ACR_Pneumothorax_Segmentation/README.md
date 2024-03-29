# SIIM-ACR Pneumothorax Segmentation  
Reference: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview  
  
## Environment and packages  
ubuntu 20.04 + cuda 11.1 + cudnn 8.0.5  
pytorch 1.8.1  

## Data preprocess   
Turn dcm files into jpg files; rle to mask into jpg files    
![alt text](https://github.com/allen050883/Project/blob/main/kaggle_SIIM-ACR_Pneumothorax_Segmentation/check_dataset/1.2.276.0.7230010.3.1.4.8323329.1909.1517875170.188918.jpg)
  
## DataLoader  
Reading all files complete path, use cv2.imwrite open images  
Data Argumentation use albumentations  
```
import albumentations as A
A.Compose([
    A.HorizontalFlip(),
    
    # The brightness and contrast might enhance the difficulty of detetmining where the area full of air and no blood vessel.
    """
    A.OneOf([
        A.RandomContrast(),
        A.RandomGamma(),
        A.RandomBrightness(),
        ], p=0.3),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
    """
    
    A.ShiftScaleRotate(),
])
```
  
## Network  
ResUnet, using pytorch resnet-50 layer for pre-train model  
Reference: https://arxiv.org/pdf/1711.10684.pdf  
  
CBAM, for attention block
Reference: https://arxiv.org/pdf/1807.06521.pdf  
  
Optimizer using Ranger in torch-optimizer package  
Paper: New Deep Learning Optimizer, Ranger: Synergistic combination of RAdam + LookAhead for the best of both (2019)  
[https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d]  
Reference Code: https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer  
  
## Loss  
score: dice_score (from offical websites)  
Combo loss: dice_loss + binary_cross_entropy_loss  
Proposed loss: focal loss - log(dice_score) # The training data of normal:abnoraml = 1:1  
Reference: https://arxiv.org/pdf/1805.02798.pdf    
  
  
# How to run the code?  
1. Install environment  
nvidia driver 460, cuda 11.1, cudnn 8.0.5. 
```
pip install -r requirements.txt
```

2. Run preprocess  
```
python preprocess.py
```
Generate 6 folder, including:  
  train_img_normal  
  train_mask_normal  
  train_img_abnormal  
  train_mask_abnormal  
  test_img  
  test_mask  
  
PS. you can use ```check_dataset.py``` to check the mask location on the right side.  
```
python check_dataset.py
```

3. Setting Config in 'utils' folder, and use pre-train weight or not  
```
python main.py
```
  
4. Result  
```
python predict.py
```
Predict the validation image, and draw in th folder.  
