# patchwork.py

import numpy as np
from config import T

def embed_patchwork(img, watermark):
    marked = img.copy()
    DIFF = np.zeros_like(img)
    
    flat_img = img.flatten()
    wm_bits = watermark.flatten()
    
    for i, bit in enumerate(wm_bits):
        i1, i2 = i * 2, i * 2 + 1
        if i2 >= len(flat_img): break
        
        if bit == 1:
            flat_img[i1] = np.clip(flat_img[i1] + T, 0, 255)
            flat_img[i2] = np.clip(flat_img[i2] - T, 0, 255)
        else:
            flat_img[i1] = np.clip(flat_img[i1] - T, 0, 255)
            flat_img[i2] = np.clip(flat_img[i2] + T, 0, 255)
    
    marked = flat_img.reshape(img.shape)
    DIFF = marked - img
    
    # Introduce small noise directly into DIFF
    noise = np.random.normal(0, 2, DIFF.shape).astype(np.uint8)
    DIFF = np.clip(DIFF + noise, -255, 255)

    return marked, DIFF
