from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,BatchNormalization
import models.CNN.cnn_config as cfg


class CNN_model(object):

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
        # model.add(Dropout(self.dropout_rate))
        model.add(
            BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                               gamma_initializer='ones', moving_mean_initializer='zeros',
                               moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                               beta_constraint=None, gamma_constraint=None))

        # conv layer 2
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same',
                         activation=self.activation, name='conv4'))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same',
                         activation=self.activation, name='conv5'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='maxpool6'))
        # model.add(Dropout(self.dropout_rate))
        model.add(
            BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                               gamma_initializer='ones', moving_mean_initializer='zeros',
                               moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                               beta_constraint=None, gamma_constraint=None))

        # fc layer
        model.add(Flatten())
        model.add(Dense(256, activation=self.activation))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        # plot_model(model, to_file=cfg.DATA_PATH + '/model.png')

        return model


