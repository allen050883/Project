import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.config import Config

config = Config()
CHECK_DATASET_DIR = config.CHECK_DATASET_DIR
os.makedirs(CHECK_DATASET_DIR, exist_ok=True)

# file list
file_list = [i.split('.jpg')[0] for i in os.listdir('train_img_abnormal')]

num = 0
for f in file_list:
    fig, (ax1, ax2, ax3) = plt.subplots( nrows=1, ncols=3 )
    img = cv2.imread('./train_img_abnormal/'+f+'.jpg')
    mask = cv2.imread('./train_mask_abnormal/'+f+'_mask.jpg')
    ax1.imshow(img, cmap=plt.cm.bone)
    ax2.imshow(mask, cmap='gray')
    
    ax3.imshow(img, cmap=plt.cm.bone)
    ax3.grid(False)

    mask = ((mask / 255.) - 1) * (-1)
    mask = np.concatenate([np.ones([1024, 1024, 1]), (mask[:,:,0]).reshape(1024, 1024, 1), np.ones([1024, 1024, 1])], axis=-1)
    mask = np.array(mask, np.float32)
    ax3.imshow(mask, alpha = 0.2)

    fig.savefig('{}/{}.jpg'.format(CHECK_DATASET_DIR, f))
    plt.close(fig)
    
    num += 1; print(num)
    if num==20: exit()
