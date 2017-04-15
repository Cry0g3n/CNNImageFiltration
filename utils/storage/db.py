import pickle


def pack_data(data, save_path, save_name):
    with open(save_path + '\\' + save_name, 'wb') as f:
        pickle.dump(data, f)


def unpack_data(save_path, save_name):
    with open(save_path + '\\' + save_name, 'rb') as f:
        data = pickle.load(f)

    return data
