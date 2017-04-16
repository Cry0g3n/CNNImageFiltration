import json
import pickle

from keras.models import load_model

with open('../config.json') as config_data:
    config = json.load(config_data)

data_storage = config['data_storage']


def pack_data(data, save_path, save_name):
    with open(save_path + '\\' + save_name, 'wb') as f:
        pickle.dump(data, f)


def unpack_data(save_path, save_name):
    with open(save_path + '\\' + save_name, 'rb') as f:
        data = pickle.load(f)

    return data


def save_model_to_storage(filename, model):
    model.save(data_storage + '\\' + filename)


def get_model_from_storage(filename):
    model = load_model(data_storage + '\\' + filename)
