import numpy as np
from keras.engine import Input, Model
from keras.layers import Convolution2D, Deconvolution2D, merge

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

    init = Input(shape=(image_size, image_size, channels))

    output_shape = (None, image_size, image_size, channels)

    level1_1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(init)
    level2_1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(level1_1)

    level2_2 = Deconvolution2D(64, 3, 3, activation='relu', output_shape=output_shape, border_mode='same')(
        level2_1)
    level2 = merge([level2_1, level2_2], mode='sum')

    level1_2 = Deconvolution2D(64, 3, 3, activation='relu', output_shape=output_shape, border_mode='same')(level2)
    level1 = merge([level1_1, level1_2], mode='sum')

    decoded = Convolution2D(channels, 5, 5, activation='linear', border_mode='same')(level1)

    model = Model(init, decoded)

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, options['batch_size'], options['epochs'], validation_split=0.2)

    return model
