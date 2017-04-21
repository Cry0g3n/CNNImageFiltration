import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d

from models.dncnn import prepare_data, unprepare_data


def get_image_patches(image, crop_size=[40, 40], stride=20):
    shape = image.shape
    patches = []
    for x in range(0, shape[0], stride):
        if x + crop_size[0] < shape[0]:
            for y in range(0, shape[1], stride):
                if y + crop_size[1] < shape[1]:
                    patches.append(image[x:x + crop_size[0], y:y + crop_size[1]])

    return patches


def sliding_2d_res_filtration(image, patch_size=40, model=None):
    image_patches = extract_patches_2d(image, [patch_size, patch_size])

    if model is not None:
        patches = prepare_data(image_patches)
        noise_maps = model.predict(patches, verbose=1)
        noise_patches = unprepare_data(noise_maps)

        f_patches = []
        for i in range(0, len(image_patches)):
            f_patch = image_patches[i] - noise_patches[i]
            f_patches.append(f_patch)

        f_patches = np.asarray(f_patches)
    else:
        f_patches = unprepare_data(image_patches)

    f_image = reconstruct_from_patches_2d(f_patches, image.shape)

    return f_image


def sliding_2d_filtration(image, patch_size=40, model=None):
    image_patches = extract_patches_2d(image, [patch_size, patch_size])

    if model is not None:
        patches = prepare_data(image_patches)
        f_patches = model.predict(patches, verbose=1)

        f_patches = np.asarray(f_patches)
        f_patches = unprepare_data(f_patches)
    else:
        f_patches = unprepare_data(image_patches)

    f_image = reconstruct_from_patches_2d(f_patches, image.shape)

    return f_image
