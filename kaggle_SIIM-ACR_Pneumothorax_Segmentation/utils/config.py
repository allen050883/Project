class Config:
    def __init__(self):
        # Deep learning hyperparameter
        self.TRAIN_TEST_SPLIT = 0.8
        self.UPSAMPLEING = True

        # Epoch from START_EPOCH to END_EPOCH
        self.START_EPOCH = 1
        self.END_EPOCH = 500

        # Learning rate
        self.LERANING_RATE = 1e-2

        # Batch size setting
        self.BATCH_SIZE_TRAINING = 72
        self.BATCH_SIZE_VALIDATION = 72
        self.BATCH_SIZE_TESTING = 1

        # Save directory
        self.MODEL_SAVE_DIR = 'model_save'
        self.PRED_SAVE_DIR = 'pred_model_epoch_356.0_train_dice_score_0.8079_val_dice_score_0.4811'

        # Model weight
        self.PRETRAIN_WEIGHT = ''
        # self.PRETRAIN_WEIGHT = 'model_epoch_53.0_train_dice_score_0.9018_val_dice_score_0.9049.pth'
        self.INFERENCE_WEIGHT = './model_save/model_epoch_356.0_train_dice_score_0.8079_val_dice_score_0.4811.pth'

        # File writer
        self.MODEL_SAVE_DIR = "model_save"
        self.LEARNING_HISTORY_FILENAME = "learning_history.csv"
