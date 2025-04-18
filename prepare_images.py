import cv2
import os

# Make sure output folder exists
os.makedirs("images", exist_ok=True)

# Lena - resize and grayscale
lena = cv2.imread("lena.png")
lena = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
lena = cv2.resize(lena, (256, 256))
cv2.imwrite("images/lena.png", lena)
print("Saved grayscale Lena as images/lena.png")

# Logo - resize and binarize
logo = cv2.imread("logo.jpg", cv2.IMREAD_GRAYSCALE)  # Updated for JPG
if logo is None:
    print("Error: Unable to load logo image. Check file path.")
else:
    logo = cv2.resize(logo, (64, 64))
    _, logo_bin = cv2.threshold(logo, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("images/logo.png", logo_bin)
    print("Saved binary logo as images/logo.png")
