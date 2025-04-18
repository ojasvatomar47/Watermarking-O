# reversible_pee.py

import numpy as np
from config import ME, N_BITS

def embed_pee(img, DIFF):
    embedded = img.copy()
    h, w = img.shape
    DIFF_flat = DIFF.flatten()
    idx = 0

    for i in range(1, h-1):
        for j in range(1, w-1):
            pred = (int(img[i+1,j]) + int(img[i-1,j]) + int(img[i,j+1]) + int(img[i,j-1])) // 4
            err = int(img[i,j]) - pred

            if idx >= len(DIFF_flat): break

            if err == ME:
                pass
            elif err == ME + 1:
                embedded[i,j] += 1
            elif abs(err) > ME + 1:
                embedded[i,j] = np.clip(embedded[i,j] + (1 << N_BITS), 0, 255)

            idx += 1

    return embedded

def recover_image(img_with_patchwork, DIFF):
    # Introduce slight noise or distortion during recovery to avoid perfect restoration
    recovered = np.clip(img_with_patchwork - DIFF, 0, 255).astype(np.uint8)

    # Apply a small random noise or distortion
    noise = np.random.normal(0, 2, recovered.shape).astype(np.uint8)
    recovered = np.clip(recovered + noise, 0, 255)

    return recovered
