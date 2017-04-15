import json

from models.dncnn import dncnn
from utils.storage.db import unpack_data


def train_dncnn(storage='gauss_noise_patches.pickle'):
    with open('../config.json') as config_data:
        config = json.load(config_data)

    data_storage = config['data_storage']
    train_data = unpack_data(data_storage, storage)

    x_train = []
    y_train = []

    for data_patch in train_data:
        x_train.append(data_patch['noise_patch'])
        y_train.append(data_patch['residual_patch'])

    options = {}

    dncnn(x_train, y_train, options)

    print('End')


if __name__ == '__main__':
    train_dncnn()
