import cv2
import numpy as np
import os

INPUT_DIR = "data/input"
TARGET_DIR = "data/target"
SOURCE_DIR = SOURCE_DIR = "data/clean_images_processed"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(TARGET_DIR, exist_ok=True)

for file in os.listdir(SOURCE_DIR):
    img = cv2.imread(os.path.join(SOURCE_DIR, file))

    if img is None:
        continue

    img = cv2.resize(img, (128, 128))

    # Save clean
    cv2.imwrite(os.path.join(TARGET_DIR, file), img)

    # -------- BETTER NOISE --------
    noisy = img.copy()

    # Light blur (NOT strong)
    noisy = cv2.GaussianBlur(noisy, (3, 3), 0)

    # Mild Gaussian noise
    noise = np.random.normal(0, 8, img.shape)
    noisy = noisy + noise

    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(INPUT_DIR, file), noisy)

print("✅ Better dataset created")