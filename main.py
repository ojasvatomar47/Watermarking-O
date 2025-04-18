# main.py

import os
import cv2
import numpy as np
from preprocessing import load_and_preprocess
from encryption import additive_secret_sharing
from patchwork import embed_patchwork
from reversible_pee import embed_pee
from decryption import detect_patchwork, recover_image
from evaluation import (
    calculate_ber, psnr,
    apply_jpeg_compression,
    apply_salt_pepper,
    apply_gaussian_noise,
    plot_ber_results
)
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

    # Step 6A: Decryption and Extraction (No attack)
    extracted_wm = detect_patchwork(pw_img, watermark.shape)
    recovered_img = recover_image(pw_img, DIFF)

    # Step 7A: Evaluation (No attack)
    ber = calculate_ber(watermark, extracted_wm)
    psnr_val = psnr(img, recovered_img)

    print(f"[RESULT] PSNR between original and recovered image: {psnr_val:.2f} dB")
    print(f"[RESULT] Bit Error Rate (BER) of watermark extraction: {ber:.4f}")

    # Step 6B: Apply Attacks
    jpeg_attacked = apply_jpeg_compression(pw_img, quality=50)
    sp_attacked = apply_salt_pepper(pw_img, amount=0.01)
    gaussian_attacked = apply_gaussian_noise(pw_img, std=15)

    # Step 7B: Extract watermark from attacked versions
    jpeg_wm = detect_patchwork(jpeg_attacked, watermark.shape)
    sp_wm = detect_patchwork(sp_attacked, watermark.shape)
    gauss_wm = detect_patchwork(gaussian_attacked, watermark.shape)

    # Step 8B: BER after attack
    ber_jpeg = calculate_ber(watermark, jpeg_wm)
    ber_sp = calculate_ber(watermark, sp_wm)
    ber_gauss = calculate_ber(watermark, gauss_wm)

    print("\n[ATTACK TESTING RESULTS]")
    print(f"JPEG Compression BER: {ber_jpeg:.4f}")
    print(f"Salt & Pepper BER:     {ber_sp:.4f}")
    print(f"Gaussian Noise BER:    {ber_gauss:.4f}")

    # Step 9: Save Results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "original.png"), img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "x1.png"), x1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "x2.png"), x2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "watermarked_patchwork.png"), pw_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "watermarked_encrypted.png"), watermarked_share)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "recovered.png"), recovered_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "extracted_watermark.png"), extracted_wm * 255)

    # Save attacked images
    cv2.imwrite(os.path.join(OUTPUT_DIR, "attacked_jpeg.png"), jpeg_attacked)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "attacked_saltpepper.png"), sp_attacked)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "attacked_gaussian.png"), gaussian_attacked)

    # Step 10: Save BER Plot
    ber_data = {
        "No Attack": ber,
        "JPEG": ber_jpeg,
        "Salt & Pepper": ber_sp,
        "Gaussian": ber_gauss
    }
    plot_ber_results(ber_data, save_path=os.path.join(OUTPUT_DIR, "ber_results.png"))

    print(f"\n[INFO] All results and plots saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
