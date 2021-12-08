class Config:
    def __init__(self):
        # Deep learning hyperparameter
        self.TRAIN_TEST_SPLIT = 0.8

        # Epoch from START_EPOCH to END_EPOCH
        self.START_EPOCH = 43
        self.END_EPOCH = 500

        # Learning rate
        self.LERANING_RATE = 0.001

        # Batch size setting
        self.BATCH_SIZE_TRAINING = 84
        self.BATCH_SIZE_VALIDATION = 84
        self.BATCH_SIZE_TESTING = 1

        # Save directory
        self.MODEL_SAVE_DIR = 'model_save'
        self.PRED_SAVE_DIR = 'pred_train_dice_score_0.7375_val_dice_score_0.6585'

        # Model weight
        self.PRETRAIN_WEIGHT = "./model_save_1208_256_2/model_epoch_433.0_train_dice_score_0.3575_val_dice_score_0.3625.pth"
        self.INFERENCE_WEIGHT = "./model_save_1205/model_epoch_197.0_train_dice_score_0.7375_val_dice_score_0.6585.pth"

        # File writer
        self.MODEL_SAVE_DIR = "model_save"
        self.LEARNING_HISTORY_FILENAME = "learning_history.csv"