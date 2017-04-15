import numpy as np


def psnr(im1, im2):
    mse = (np.abs(im1 - im2) ** 2).sum() / (im1.shape[0] * im1.shape[1])
    result = 10 * np.log10(255 * 255 / mse)
    return result
