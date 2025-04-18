# evaluation.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_ber(original, extracted):
    return np.sum(original != extracted) / original.size

def psnr(original, recovered):
    mse = np.mean((original - recovered) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def apply_jpeg_compression(img, quality=50):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 0)

def apply_salt_pepper(img, amount=0.01):
    noisy = img.copy()
    num_salt = int(np.ceil(amount * img.size * 0.5))
    num_pepper = int(np.ceil(amount * img.size * 0.5))

    # Salt
    coords = [np.random.randint(0, i, num_salt) for i in img.shape]
    noisy[coords[0], coords[1]] = 255

    # Pepper
    coords = [np.random.randint(0, i, num_pepper) for i in img.shape]
    noisy[coords[0], coords[1]] = 0

    return noisy

def apply_gaussian_noise(img, mean=0, std=10):
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

def plot_ber_results(ber_dict, save_path=None):
    attack_types = list(ber_dict.keys())
    values = list(ber_dict.values())

    plt.figure(figsize=(8, 5))
    plt.bar(attack_types, values, color='teal')
    plt.title("Bit Error Rate (BER) under Different Attacks")
    plt.xlabel("Attack Type")
    plt.ylabel("BER")
    plt.ylim(0, 1)

    for i, val in enumerate(values):
        plt.text(i, val + 0.02, f"{val:.4f}", ha='center')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
