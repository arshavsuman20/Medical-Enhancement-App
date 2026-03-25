import cv2
import os

SOURCE_DIR = "data/clean_images"
CLEAN_DIR = "data/clean_images_processed"

os.makedirs(CLEAN_DIR, exist_ok=True)

count = 0

for root, dirs, files in os.walk(SOURCE_DIR):
    for file in files:
        path = os.path.join(root, file)

        img = cv2.imread(path)

        if img is None:
            continue

        # Resize
        img = cv2.resize(img, (128, 128))

        # Optional: convert to grayscale (better for MRI)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        save_path = os.path.join(CLEAN_DIR, f"{count}.png")
        cv2.imwrite(save_path, img)

        count += 1

print(f"✅ Clean dataset ready: {count} images")