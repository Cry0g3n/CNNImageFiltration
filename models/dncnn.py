import numpy as np
from keras.layers import Conv2D
from keras.models import Sequential

image_size = 40
channels = 1  # TODO: Автоматическое определение количества каналов


def prepare_data(data, normalize=True):
    data = np.asarray(data)
    if normalize:
        data = data.astype('float32') / 255
    data = np.reshape(data, (len(data), image_size, image_size, channels))
    return data


def dncnn(x_train, y_train, options):
    x_train = prepare_data(x_train)
    y_train = prepare_data(y_train)

    model = Sequential()
    model.add(Conv2D(64, [3, 3], padding='same', activation='relu', use_bias=False,
                     input_shape=[image_size, image_size, channels]))  # layer 1

    model.add(Conv2D(64, [3, 3], padding='same', activation='relu'))  # layer 2
    model.add(Conv2D(64, [3, 3], padding='same', activation='relu'))  # layer 3
    model.add(Conv2D(64, [3, 3], padding='same', activation='relu'))  # layer 4
    model.add(Conv2D(64, [3, 3], padding='same', activation='relu'))  # layer 5
    model.add(Conv2D(64, [3, 3], padding='same', activation='relu'))  # layer 6
    model.add(Conv2D(64, [3, 3], padding='same', activation='relu'))  # layer 7
    model.add(Conv2D(64, [3, 3], padding='same', activation='relu'))  # layer 8
    model.add(Conv2D(64, [3, 3], padding='same', activation='relu'))  # layer 9
    model.add(Conv2D(64, [3, 3], padding='same', activation='relu'))  # layer 10
    model.add(Conv2D(64, [3, 3], padding='same', activation='relu'))  # layer 11
    model.add(Conv2D(64, [3, 3], padding='same', activation='relu'))  # layer 12
    model.add(Conv2D(64, [3, 3], padding='same', activation='relu'))  # layer 13
    model.add(Conv2D(64, [3, 3], padding='same', activation='relu'))  # layer 14
    model.add(Conv2D(64, [3, 3], padding='same', activation='relu'))  # layer 15
    model.add(Conv2D(64, [3, 3], padding='same', activation='relu'))  # layer 16

    model.add(Conv2D(channels, [3, 3], padding='same', activation='relu'))  # layer 3

    model.compile(optimizer='sgd', loss='mean_squared_error')

    model.fit(x_train, y_train, batch_size=128, epochs=1)
