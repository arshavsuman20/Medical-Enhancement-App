from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

IMG_SIZE = 128

model = load_model("models/unet.h5")

img = cv2.imread("data/sample_input/input.png", cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0

img = np.expand_dims(img, axis=(0, -1))

output = model.predict(img)

output = (output[0] * 255).astype(np.uint8)

os.makedirs("data/sample_output", exist_ok=True)
cv2.imwrite("data/sample_output/output.png", output)

print("✅ Output saved at data/sample_output/output.png")