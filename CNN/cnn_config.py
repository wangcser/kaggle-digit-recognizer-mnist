import os


# data path

DATA_PATH = '/media/super/Dev Data/ml_data_set/Kaggle_MNIST'
DATA_CACHE_PATH = os.path.join(DATA_PATH, 'cache')
WEIGHT_PATH = os.path.join(DATA_PATH, 'weight')
RESULT_PATH = os.path.join(DATA_PATH, 'result')

# net parameters

EPOCHS = 300
BATCH_SIZE = 64

# tf log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

