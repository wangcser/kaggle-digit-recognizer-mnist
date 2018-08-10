import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model

from data_set.data_pre_process import Load_full_data, use_comp_test
from models.simple_CNN.simple_cnn_model import NN_model
import models.simple_CNN.simple_cnn_config as cfg


class Solver(object):

    def __init__(self, model, data):
        # data
        self.data = data
        self.x_train, self.y_train, self.x_val, self.y_val = self.data.load_mnist('train', split_val=True)
        # model
        self.model = model.build()
        self.epochs = cfg.EPOCHS
        self.batch_size = cfg.BATCH_SIZE
        self.optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        # set the call back methods, check_point can store the net cfg and weight.
        self.tensorboard = TensorBoard(log_dir=cfg.LOG_PATH, histogram_freq=0, write_graph=True, write_images=True)
        self.check_point = ModelCheckpoint(cfg.WEIGHT_FILE, monitor='val_loss', verbose=0,
                                           save_best_only=True, save_weights_only=True, mode='auto', period=10)
        # predict
        self.test = use_comp_test()
        self.result_file = cfg.RESULT_FILE
        # data argumentation
        self.data_argumentor = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False
        )

    def train(self, mode='First'):
        # compile the compute graph.
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        if mode is 'continue':
            self.model = load_model(cfg.WEIGHT_PATH + 'cnn_weight.h5')

        # set a learning rate annealer
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3,
                                                    verbose=1, factor=0.5, min_lr=0.00001)
        # Fit the model
        # 1. prepare the data.
        self.data_argumentor.fit(self.x_train)
        call_back_list = [self.check_point, self.tensorboard, learning_rate_reduction]
        # 2. feed data to the model with generator method.
        self.history = self.model.fit_generator(
            self.data_argumentor.flow(self.x_train, self.y_train, batch_size=self.batch_size),
            epochs=self.epochs,
            validation_data=(self.x_val, self.y_val),
            verbose=2,
            steps_per_epoch=self.x_train.shape[0] // self.batch_size,
            callbacks=call_back_list
        )

    def predict(self):
        self.results = self.model.predict(use_comp_test())

        # select the index with the max probability
        self.results = np.argmax(self.results, axis=1)
        self.results = pd.Series(self.results, name="Label")

        submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), self.results], axis=1)
        submission.to_csv(self.result_file, index=False)

    def evaluate(self):
        # evaluate the model
        # plot the loss and accuracy curves for training and validation
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(self.history.history['loss'], color='b', label="Training loss")
        ax[0].plot(self.history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
        legend = ax[0].legend(loc="best", shadow=True)

        ax[1].plot(self.history.history['acc'], color='b', label="Training accuracy")
        ax[1].plot(self.history.history['val_acc'], color='r', label="validation accuracy")
        legend = ax[1].legend(loc='best', shadow=True)

        plt.show()


def main():
    model = NN_model()
    # data_set = Load_data()
    data_set = Load_full_data()

    solver = Solver(model, data_set)

    solver.train()

    solver.predict()

    solver.evaluate()


if __name__ == '__main__':
    main()
