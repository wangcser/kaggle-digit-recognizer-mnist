#
# visualize the data set.
#

import matplotlib.pyplot as plt
from data_set.data_transform import Load_data, Load_full_data


def main():
    data = Load_data()
    X_train, y_train, X_val, y_val = data.load_mnist('train')
    test = data.load_mnist('test')
    # print(X_train.shape, y_train.shape, X_val.shape, test.shape)

    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()

    for i in range(10):
        img = X_train[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()