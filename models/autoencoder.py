import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.models import Sequential

image_size = 64
channels = 1  # TODO: Автоматическое определение количества каналов


def prepare_data(data, normalize=True):
    data = np.asarray(data)
    if normalize:
        data = data.astype('float32') / 255
    data = np.reshape(data, (len(data), image_size, image_size, channels))
    return data


def unprepare_data(data):  # TODO: Подумать над названием
    data = data.astype('float32')
    result = []
    for i in range(0, data.shape[0]):
        result.append(data[i, :, :, 0] * 255)

    result = np.asarray(result)
    return result


def autoencoder(x_train, y_train, options):
    x_train = prepare_data(x_train)
    y_train = prepare_data(y_train)

    model = Sequential()
    model.add(Conv2D(64, [4, 4], padding='same', activation='relu', use_bias=False,
                     input_shape=[image_size, image_size, channels]))  # layer 1
    model.add(MaxPooling2D(padding='same'))  # layer 1

    model.add(Conv2D(64, [4, 4], padding='same'))  # layer 2
    model.add(MaxPooling2D(padding='same'))  # layer 2

    model.add(Conv2D(128, [4, 4], padding='same'))  # layer 3
    model.add(MaxPooling2D(padding='same'))  # layer 3

    model.add(Conv2D(256, [4, 4], padding='same'))  # layer 4
    model.add(MaxPooling2D(padding='same'))  # layer 4

    model.add(Conv2D(512, [4, 4], padding='same'))  # layer 5

    model.add(Conv2DTranspose(256, [4, 4], padding='same'))  # layer 6
    model.add(UpSampling2D())  # layer 6

    model.add(Conv2DTranspose(128, [4, 4], padding='same'))  # layer 7
    model.add(UpSampling2D())  # layer 7

    model.add(Conv2DTranspose(64, [4, 4], padding='same'))  # layer 8
    model.add(UpSampling2D())  # layer 8

    model.add(Conv2DTranspose(64, [4, 4], padding='same'))  # layer 9
    model.add(UpSampling2D())  # layer 9

    model.add(Conv2DTranspose(channels, [4, 4], padding='same', activation='linear'))  # layer 10

    model.compile(optimizer='sgd', loss='mean_squared_error')

    model.fit(x_train, y_train, options['batch_size'], options['epochs'], validation_split=0.2)

    return model
