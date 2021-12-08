import os
import pandas as pd
import matplotlib.pyplot as plt

# Hyperparameter
MODEL_SAVE_FOLDER = 'model_save_1207_loss_not_good'
LEARNING_CURVE = 'learning_history_1207_loss_not_good.jpg'

# Read files
model_list = os.listdir('./'+MODEL_SAVE_FOLDER+'/')
model_list.sort()

# Clear file name
epoch = []; train_dice_score = []; val_dice_score = []
for name in model_list:
    epoch.append( int(float((name.split('model_epoch_')[1]).split('_train_dice_score')[0])) )
    train_dice_score.append( float((name.split('_train_dice_score_')[1]).split('_val_dice_score')[0]) )
    val_dice_score.append( float((name.split('_val_dice_score_')[1]).split('.pth')[0]) )


# # Hyperparameter
# CSV_FILE = "learning_history_1205.csv"

# # Read learning curve csv file
# csv_file = pd.read_csv(CSV_FILE)

# # Take Data Column
# epoch = list(csv_file['epoch'])
# train_dice_score = list(csv_file['train_dice_score'])
# val_dice_score = list(csv_file['val_dice_score'])

# Draw
plt.plot(epoch, train_dice_score, color='blue', label='train_dice_score')
plt.plot(epoch, val_dice_score, color='red', label='val_dice_score')
plt.title("Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Dice score")
plt.legend()

# plt.savefig(CSV_FILE.split('csv')[0] + 'jpg')
plt.savefig(LEARNING_CURVE)
plt.close()

