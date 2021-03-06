import json
from cv2 import imwrite, imread, cvtColor, COLOR_RGB2GRAY

from models.distortions import awgn
from utils.image_utils import get_image_patches
from utils.storage.db import pack_data
from utils.storage.image import get_image_filename_list, load_image_filename_list


def prepare_awgn_data(filename='gauss_noise_patches.pickle', sigma=[15, 25, 50]):
    data_list = []

    with open('../config.json') as config_data:
        config = json.load(config_data)

    dataset_path = config['dataset_path']
    train_dataset_path = dataset_path + '\\' + config['train_dataset']
    data_storage = config['data_storage']

    files = get_image_filename_list(train_dataset_path)
    images = load_image_filename_list(files, gray=True)

    for image in images:
        clear_patches = get_image_patches(image, crop_size=[64, 64])
        for s in sigma:
            noise_image = awgn(image, s)
            imwrite('temp_noise.bmp', noise_image)
            noise_image = imread('temp_noise.bmp')
            noise_image = cvtColor(noise_image, COLOR_RGB2GRAY)
            noise_patches = get_image_patches(noise_image, crop_size=[64, 64])

            data_list.extend(prepare_data_patch(clear_patches, noise_patches))

    pack_data(data_list, data_storage, filename)


def prepare_data_patch(clear_patches, noise_patches):
    data_patch = []
    for i in range(0, len(clear_patches)):
        clear_patch = clear_patches[i]
        noise_patch = noise_patches[i]

        data_patch.append({
            'noise_patch': noise_patch,
            'clear_patch': clear_patch,
        })

    return data_patch


if __name__ == '__main__':
    prepare_awgn_data(filename='gauss_noise_patches-size-64.pickle')
