# evaluation.py

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_ber(original, extracted):
    return np.sum(original != extracted) / original.size

def apply_jpeg_compression(img, quality=50):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 0)

def apply_salt_pepper(img, amount=0.05):
    output = img.copy()
    num_salt = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    output[coords] = 255
    return output

def psnr(original, recovered):
    mse = np.mean((original - recovered) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))
