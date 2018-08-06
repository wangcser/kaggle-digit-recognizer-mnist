import os
import data_set.data_config as data_cfg


# net utils path
DATA_CACHE_PATH = os.path.join(data_cfg.DATA_SET_PATH, 'data_cache')
RESULT_PATH = os.path.join(data_cfg.DATA_SET_PATH, 'result')
WEIGHT_PATH = os.path.join(data_cfg.DATA_SET_PATH, 'weight')
CHECK_POINT_PATH = os.path.join(WEIGHT_PATH, 'check_point')

WEIGHT_FILE = os.path.join(CHECK_POINT_PATH, "epoch_{epoch:002d}-valAcc_{val_acc:.4f}.hdf5")
RESULT_FILE = os.path.join(RESULT_PATH, "cnn_result.csv")

LOG_PATH = os.path.join(data_cfg.DATA_SET_PATH, 'logs')
# net parameters
DROPOUT_RATE = 0.25
ACTIVATION = 'relu'
EPOCHS = 100
BATCH_SIZE = 64

# tf log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

