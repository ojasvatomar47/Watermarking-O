# decryption.py

import numpy as np
from config import T

def detect_patchwork(image, watermark_shape):
    wm = []
    flat = image.flatten()
    
    for i in range(0, len(flat), 2):
        if i+1 >= len(flat): break
        if flat[i] > flat[i+1]:
            wm.append(1)
        else:
            wm.append(0)

    expected_size = watermark_shape[0] * watermark_shape[1]
    wm = wm[:expected_size]
    
    return np.array(wm).reshape(watermark_shape)

def recover_image(img_with_patchwork, DIFF):
    return np.clip(img_with_patchwork - DIFF, 0, 255).astype(np.uint8)
