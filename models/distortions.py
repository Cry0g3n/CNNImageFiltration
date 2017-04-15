from skimage.util import random_noise


def awgn(image, sigma):  # TODO: Проверить
    variance = sigma * sigma
    noise_image = random_noise(image, mode='gaussian', var=variance)
    return noise_image
