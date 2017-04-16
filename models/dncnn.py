import numpy as np
from keras.layers import Conv2D, SeparableConv2D, BatchNormalization, Activation
from keras.models import Sequential

image_size = 40
channels = 1  # TODO: Автоматическое определение количества каналов


def prepare_data(data, normalize=True):
    data = np.asarray(data)
    if normalize:
        data = data.astype('float32') / 255
    data = np.reshape(data, (len(data), image_size, image_size, channels))
    return data


def add_conv2d_bn_relu_layer(model):
    model.add(SeparableConv2D(64, [3, 3], padding='same'))  # layer 2
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    return model


def dncnn(x_train, y_train, options):
    x_train = prepare_data(x_train)
    y_train = prepare_data(y_train)

    model = Sequential()
    model.add(Conv2D(64, [3, 3], padding='same', activation='relu', use_bias=False,
                     input_shape=[image_size, image_size, channels]))  # layer 1

    for i in range(0, 15):
        model = add_conv2d_bn_relu_layer(model)  # layer 2-16

    model.add(Conv2D(channels, [3, 3], padding='same', activation='linear'))  # layer 17

    model.compile(optimizer='sgd', loss='mean_squared_error')

    model.fit(x_train, y_train, batch_size=32, epochs=1)

    return model
