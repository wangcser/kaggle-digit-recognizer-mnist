from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
import models.simple_CNN.simple_cnn_config as cfg


class NN_model(object):

    def __init__(self):
        # use cfg to init the net args.
        self.activation = cfg.ACTIVATION
        self.dropout_rate = cfg.DROPOUT_RATE

    def build(self):
        model = Sequential()

        # conv layer 1
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                         activation=self.activation, input_shape=(28, 28, 1), name='conv1'))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                         activation=self.activation, name='conv2'))
        model.add(MaxPool2D(pool_size=(2, 2), name='maxpool3'))

        # conv layer 2
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                         activation=self.activation, name='conv4'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                         activation=self.activation, name='conv5'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='maxpool6'))

        # fc layer 3
        model.add(Flatten())
        model.add(Dense(256, activation=self.activation))
        # output
        model.add(Dense(10, activation='softmax'))

        return model


