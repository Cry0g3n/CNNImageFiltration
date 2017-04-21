import json

from models.autoencoder import autoencoder
from models.dncnn import dncnn
from utils.storage.db import unpack_data, save_model_to_storage


def train_autoencoder(storage='gauss_noise_patches-size-64.pickle'):
    with open('../config.json') as config_data:
        config = json.load(config_data)

    data_storage = config['data_storage']
    train_data = unpack_data(data_storage, storage)

    x_train = []
    y_train = []

    for data_patch in train_data:
        x_train.append(data_patch['noise_patch'])
        y_train.append(data_patch['clear_patch'])

    options = {
        'batch_size': 128,
        'epochs': 10
    }

    net = autoencoder(x_train, y_train, options)
    save_model_to_storage('autoencoder.h5', net)

    print('End')


if __name__ == '__main__':
    train_autoencoder()
