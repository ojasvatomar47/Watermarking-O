# preprocessing.py

import cv2
import numpy as np
from config import IMAGE_SIZE

def load_and_preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMAGE_SIZE)
    return img
