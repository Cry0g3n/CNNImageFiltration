import json

import numpy as np
from keras import callbacks
from keras.engine import Input
from keras.models import load_model

from utils.storage.db import unpack_data

with open('../config.json') as config_data:
    config = json.load(config_data)

data_storage = config['data_storage']


class BaseSuperResolutionModel(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.x_data = None
        self.y_data = None
        self.raw_data = None
        self.height = None
        self.width = None
        self.channels = None
        self.model_path = data_storage + '\\' + model_name + '.h5'

    def create_model(self, height=32, width=32, channels=3, batch_size=128):
        shape = (width, height, channels)
        self.height = height
        self.width = width
        self.channels = channels

        init = Input(shape=shape)

        return init

    def fit(self, batch_size=128, nb_epochs=100):
        if self.model is None:
            self.model = self.create_model(batch_size=batch_size)

        callback_list = [callbacks.ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True,
                                                   mode='max', save_weights_only=False)]

        print("Training model : %s" % self.__class__.__name__)
        self.model.fit(self.x_data, self.y_data, batch_size=batch_size, epochs=nb_epochs, validation_split=0.2,
                       callbacks=callback_list)

    def load_data(self, filename):
        self.raw_data = unpack_data(data_storage, filename)

    def prepare_data(self, data, normalize_flag=True):
        data = np.asarray(data)

        if normalize_flag:
            data = data.astype('float32') / 255

        data = np.reshape(data, (len(data), self.height, self.width, self.channels))
        return data

    @staticmethod
    def extend_data(data):
        data = data.astype('float32')
        result = []
        for i in range(0, data.shape[0]):
            result.append(data[i, :, :, 0] * 255)

        result = np.asarray(result)
        return result

    def save_data(self, x_name, y_name):
        train_data = self.raw_data

        x_train = []
        y_train = []

        for data_patch in train_data:
            x_train.append(data_patch[x_name])
            y_train.append(data_patch[y_name])

        self.x_data = self.prepare_data(x_train)
        self.y_data = self.prepare_data(y_train)

    def save_model(self, filename):
        self.model.save(data_storage + '\\' + filename)

    def load_model(self, filename):
        self.model = load_model(data_storage + '\\' + filename)

    def predict(self, image, stride=32):
        shape = image.shape
        f_image = np.copy(image)
        for x in range(0, shape[0], stride):
            if x + self.height <= shape[0]:
                for y in range(0, shape[1], stride):
                    if y + self.width <= shape[1]:
                        patches = [image[x:x + self.height, y:y + self.width]]
                        data = self.prepare_data(patches)
                        f_image[x:x + self.height, y:y + self.width] = \
                            self.extend_data(self.model.predict_on_batch(data))[
                                0]

        return f_image
