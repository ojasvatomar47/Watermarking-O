# encryption.py

import numpy as np
from utils import scramble_blocks
from config import N_BITS

def additive_secret_sharing(image):
    hsb = image >> N_BITS
    lsb = image & ((1 << N_BITS) - 1)

    x1_hsb = np.random.randint(0, 1 << (8 - N_BITS), image.shape)
    x1_lsb = np.random.randint(0, 1 << N_BITS, image.shape)

    x2_hsb = (hsb - x1_hsb) % (1 << (8 - N_BITS))
    x2_lsb = (lsb - x1_lsb) % (1 << N_BITS)

    x1 = (x1_hsb << N_BITS) + x1_lsb
    x2 = (x2_hsb << N_BITS) + x2_lsb

    x1 = scramble_blocks(x1)
    x2 = scramble_blocks(x2)

    return x1, x2
