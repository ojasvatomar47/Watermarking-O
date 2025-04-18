# main.py

import os
import cv2
import numpy as np
from preprocessing import load_and_preprocess
from encryption import additive_secret_sharing
from patchwork import embed_patchwork
from reversible_pee import embed_pee
from decryption import detect_patchwork, recover_image
from evaluation import calculate_ber, psnr
from config import *

def main():
    # Step 1: Load and preprocess the image
    img = load_and_preprocess(INPUT_IMAGE_PATH)
    watermark_raw = load_and_preprocess(WATERMARK_PATH)

    # Step 2: Resize watermark to fit embedding capacity
    wm_target_shape = (IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2)
    watermark = cv2.resize(watermark_raw, wm_target_shape)
    _, watermark = cv2.threshold(watermark, 127, 1, cv2.THRESH_BINARY)  # binary (0 or 1)

    print(f"[INFO] Image shape: {img.shape}")
    print(f"[INFO] Watermark shape (after resize): {watermark.shape}")

    # Step 3: Encryption via additive secret sharing
    x1, x2 = additive_secret_sharing(img)

    # Step 4: Stage 1 - Robust watermarking (Patchwork)
    pw_img, DIFF = embed_patchwork(img, watermark)

    # Step 5: Stage 2 - Reversible watermarking (PEE)
    watermarked_share = embed_pee(x1, DIFF)

    # Step 6: Decryption and Extraction
    extracted_wm = detect_patchwork(pw_img, watermark.shape)
    recovered_img = recover_image(pw_img, DIFF)

    # Step 7: Evaluation Metrics
    ber = calculate_ber(watermark, extracted_wm)
    psnr_val = psnr(img, recovered_img)

    print(f"[RESULT] PSNR between original and recovered image: {psnr_val:.2f} dB")
    print(f"[RESULT] Bit Error Rate (BER) of watermark extraction: {ber:.4f}")

    # Step 8: Save Results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "original.png"), img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "x1.png"), x1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "x2.png"), x2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "watermarked_patchwork.png"), pw_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "watermarked_encrypted.png"), watermarked_share)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "recovered.png"), recovered_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "extracted_watermark.png"), extracted_wm * 255)  # visualize binary wm

    print(f"[INFO] All results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
