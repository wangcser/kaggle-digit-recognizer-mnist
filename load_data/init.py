import pandas as pd

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

import CNN.cnn_config as cfg


class Load_data(object):

    def __init__(self):

        # load data as pandas format
        self.data_path = cfg.DATA_PATH
        self.train = pd.read_csv(self.data_path + "/train.csv")
        self.test = pd.read_csv(self.data_path + "/test.csv")

    def format_train_data(self):

        y_train = self.train["label"]

        # drop 'label' column
        x_train = self.train.drop(labels=["label"], axis=1)

        # free some space
        del self.train

        # normalize the data
        x_train = x_train / 255.0


        # reshape image in 3-d: 28*28*1
        x_train = x_train.values.reshape(-1, 28, 28, 1)


        # encode label to one-hot vector
        y_train = to_categorical(y_train, num_classes = 10)

        # split training and validation set
        # ser the random seed
        random_seed = 2
        # split
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=random_seed)

        return x_train, y_train, x_val, y_val

    def format_test_data(self):

        test = self.test / 255.0
        test = test.values.reshape(-1, 28, 28, 1)

        return test
