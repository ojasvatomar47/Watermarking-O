# utils.py

import numpy as np

def scramble_blocks(img, block_size=8):
    h, w = img.shape
    blocks = [(i, j) for i in range(0, h, block_size) for j in range(0, w, block_size)]
    np.random.shuffle(blocks)
    scrambled = img.copy()

    for idx, (i, j) in enumerate(blocks):
        i2, j2 = blocks[idx]
        scrambled[i:i+block_size, j:j+block_size] = img[i2:i2+block_size, j2:j2+block_size]

    return scrambled
