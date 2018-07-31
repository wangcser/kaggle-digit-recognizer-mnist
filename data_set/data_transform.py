#
# format the raw data to train set and test set.
#

import os
import struct
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import data_set.data_config as cfg


class Load_data(object):
    #
    # there are 42,000 train samples, spilt it into 90% train set(37,800) and 10% val set(4,200)
    # the image data X format in tensor as size*28*28*1, all value remapping for 0~255 to 0~1
    # the label data y format in one-hot vector as size*10
    # there are 28,000 test samples
    # sum up in 70,000 sample.
    #

    def __init__(self):
        # load data as pandas format - df
        self.data_path = cfg.DATA_PATH

    def load_mnist(self, kind='train'):
        if kind == 'train':
            train = pd.read_csv(self.data_path + "/train.csv")

            y_train = train["label"]

            # drop 'label' column
            X_train = train.drop(labels=["label"], axis=1)
            del train   # free space

            # normalize the data
            X_train = X_train / 255.0

            # reshape image in 3-d: 28*28*1
            X_train = X_train.values.reshape(-1, 28, 28, 1)

            # encode label to one-hot vector
            y_train = to_categorical(y_train, num_classes=10)

            # split training and validation set

            random_seed = 5
            # split
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_seed)

            return X_train, y_train, X_val, y_val

        else:
            test = pd.read_csv(self.data_path + "/test.csv")
            test = test / 255.0
            test = test.values.reshape(-1, 28, 28, 1)

            return test


class Load_full_data(object):
    #
    # there are 60,000 train samples, spilt it into 90% train set(54,000) and 10% val set(6,000)
    # the image data X format in tensor as size*28*28*1, all value remapping for 0~255 to 0~1
    # the label data y format in one-hot vector as size*10
    # there are 10,000 test samples
    # sum up in 70,000 samples.
    #

    def __init__(self):
        self.data_path = cfg.FULL_DATA_PATH
        # train data path
        self.train_images_path = os.path.join(self.data_path, 'train-images.idx3-ubyte')
        self.train_labels_path = os.path.join(self.data_path, 'train-labels.idx1-ubyte')
        # test data path
        self.test_images_path = os.path.join(self.data_path, 't10k-images.idx3-ubyte')
        self.test_labels_path = os.path.join(self.data_path, 't10k-labels.idx1-ubyte')

    def load_mnist(self, kind='train'):
        if kind == 'train':
            with open(self.train_labels_path, 'rb') as lbpath:
                magic, n = struct.unpack('>II', lbpath.read(8))
                y_train = np.fromfile(lbpath, dtype=np.uint8)

            with open(self.train_images_path, 'rb') as imgpath:
                magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
                X_train = np.fromfile(imgpath, dtype=np.uint8).reshape(len(y_train), 784)

            # normalize the data
            X_train = X_train / 255.0

            # reshape image in 3-d: 28*28*1
            X_train = X_train.reshape(-1, 28, 28, 1)

            # encode label to one-hot vector
            y_train = to_categorical(y_train, num_classes=10)

            # split training and validation set
            random_seed = 2
            # split
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_seed)

            return X_train, y_train, X_val, y_val

        else:
            # use the test label when cal the accurate.
            with open(self.test_labels_path, 'rb') as lbpath:
                magic, n = struct.unpack('>II', lbpath.read(8))
                test_labels = np.fromfile(lbpath, dtype=np.uint8)

            with open(self.test_images_path, 'rb') as imgpath:
                magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
                test = np.fromfile(imgpath, dtype=np.uint8).reshape(len(test_labels), 784)

            test = test / 255.0
            test = test.reshape(-1, 28, 28, 1)

            # return np.ndarray
            # print(images.shape, labels.size)
            return test


def main():

    data = Load_data()
    X_train, y_train, X_val, y_val = data.load_mnist('train')
    test = data.load_mnist('test')
    print(X_train.shape, y_train.shape, X_val.shape, test.shape)

    full_data = Load_full_data()
    X_train, y_train, X_val, y_val = full_data.load_mnist('train')
    test = full_data.load_mnist('test')
    print(X_train.shape, y_train.shape, X_val.shape, test.shape)


if __name__ == '__main__':
    main()
