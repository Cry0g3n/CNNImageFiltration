import cv2
import numpy as np
from numpy.matlib import randn


def awgn(image, sigma):
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = np.asarray(image).astype('double')
    shape = image.shape
    noise_map = randn(shape[0], shape[1])
    noise = sigma * noise_map
    noise_image = image + noise

    return noise_image
