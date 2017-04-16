import numpy as np
from skimage.measure import compare_ssim


def psnr(im1, im2):
    mse = (np.abs(im1 - im2) ** 2).sum() / (im1.shape[0] * im1.shape[1])
    result = 10 * np.log10(255 * 255 / mse)
    return result


def ssim(im1, im2):
    return compare_ssim(im1, im2)
