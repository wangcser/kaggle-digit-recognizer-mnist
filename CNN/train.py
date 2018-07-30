import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from load_data.init import Load_data
from CNN.cnn_model import CNN_model
import CNN.cnn_config as cfg


class Solver(object):

    def __init__(self, model, data):

        self.data = data
        self.x_train, self.y_train, self.x_val, self.y_val = self.data.format_train_data()
        self.model = model
        self.epochs = cfg.EPOCHS
        self.batch_size = cfg.BATCH_SIZE

        self.test = self.data.format_test_data()
        self.result_path = cfg.RESULT_PATH

        # data argumentation
        self.datagen = ImageDataGenerator(
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
            vertical_flip=False)

        self.datagen.fit(self.x_train)

    def train(self):

        # define the optimizer
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

        # compile the model
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                           metrics=['accuracy'])

        # set a learning rate annealer
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3,
                                                    verbose=1, factor=0.5, min_lr=0.00001)
        # Fit the model
        history = self.model.fit_generator(self.datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size),
                                           epochs=self.epochs, validation_data=(self.x_val, self.y_val),
                                           verbose=2, steps_per_epoch=self.x_train.shape[0] // self.batch_size,
                                           callbacks=[learning_rate_reduction])

        self.results = self.model.predict(self.test)

        # select the index with the max probability
        self.results = np.argmax(self.results, axis=1)
        self.results = pd.Series(self.results, name="Label")

        submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), self.results], axis=1)
        submission.to_csv(self.result_path + '/cnn_result.csv', index=False)

    def evaluate(self):

        # evaluate the model
        # plot the loss and accuracy curves for training and validation
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(self.history.history['loss'], color='b', label="Training loss")
        ax[0].plot(self.history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
        legend=ax[0].legend(loc="best", shadow=True)

        ax[1].plot(self.history.history['acc'], color='b', label="Training accuracy")
        ax[1].plot(self.history.history['val_acc'], color='r', label="validation accuracy")
        legend = ax[1].legend(loc='best', shadow=True)

        plt.show()

    def test(self):
        pass



def main():

    model = CNN_model()
    data_set = Load_data()

    solver = Solver(model.build(), data_set)
    solver.train()


if __name__ == '__main__':
    main()
