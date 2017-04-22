import cv2

from metrics.image_quality_metrics import ssim, psnr
from models.distortions import awgn
from utils.image_utils import sliding_2d_res_filtration, sliding_2d_filtration, grid_2d_filtration
from utils.storage.db import get_model_from_storage

# image = cv2.imread('D:\\Repositories\\CNNImageFiltration\\data\\datasets\\classic5\\peppers.bmp')
image = cv2.imread('clear.bmp')
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# cv2.imwrite('clear_mini.bmp', image)
cv2.imwrite('noise.bmp', awgn(image, 15))
noise_image = cv2.imread('noise.bmp')
noise_image = cv2.cvtColor(noise_image, cv2.COLOR_RGB2GRAY)

print('psnr = ', psnr(image, noise_image))
print('ssim = ', ssim(image, noise_image))

net = get_model_from_storage('autoencoder.h5')
# net = None

f_image = grid_2d_filtration(noise_image, patch_size=512, stride=1, model=net)
cv2.imwrite('filter.bmp', f_image)
f_image = cv2.imread('filter.bmp')
f_image = cv2.cvtColor(f_image, cv2.COLOR_RGB2GRAY)

print('psnr = ', psnr(image, f_image))
print('ssim = ', ssim(image, f_image))

print('end')
