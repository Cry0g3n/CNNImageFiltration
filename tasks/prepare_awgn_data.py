import json

from skimage import img_as_ubyte

from models.distortions import awgn
from utils.image_utils import get_image_patches
from utils.storage.db import pack_data
from utils.storage.image import get_image_filename_list, load_image_filename_list


def prepare_awgn_data(sigma=[15, 25, 50]):
    data_list = []

    with open('../config.json') as config_data:
        config = json.load(config_data)

    dataset_path = config['dataset_path']
    train_dataset_path = dataset_path + '\\' + config['train_dataset']
    data_storage = config['data_storage']

    files = get_image_filename_list(train_dataset_path)
    images = load_image_filename_list(files)

    for image in images:
        clear_patches = get_image_patches(image)
        for s in sigma:
            noise_image = awgn(image, s)
            noise_image = img_as_ubyte(noise_image)
            noise_patches = get_image_patches(noise_image)

            data_list.extend(prepare_data_patch(clear_patches, noise_patches))

    pack_data(data_list, data_storage, 'gauss_noise_patches.pickle')
    print('End')


def prepare_data_patch(clear_patches, noise_patches):
    data_patch = []
    for i in range(0, len(clear_patches)):
        clear_patch = clear_patches[0]
        noise_patch = noise_patches[0]
        residual_patch = noise_patch - clear_patch

        data_patch.append({
            'noise_patch': noise_patch,
            'residual_patch': residual_patch
        })

    return data_patch


if __name__ == '__main__':
    prepare_awgn_data()
