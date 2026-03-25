import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import os

IMG_SIZE = 128

def build_unet():
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))

    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    b = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)

    u1 = layers.UpSampling2D()(b)
    u1 = layers.Concatenate()([u1, c2])
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D()(c3)
    u2 = layers.Concatenate()([u2, c1])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)

    outputs = layers.Conv2D(3, 1, activation='sigmoid')(c4)

    return models.Model(inputs, outputs)


def load_data(input_dir, target_dir):
    X, Y = [], []

    for file in os.listdir(input_dir):
        inp = cv2.imread(os.path.join(input_dir, file))
        tar = cv2.imread(os.path.join(target_dir, file))

        if inp is None or tar is None:
            continue

        inp = cv2.resize(inp, (128, 128))
        tar = cv2.resize(tar, (128, 128))

        inp = inp / 255.0
        tar = tar / 255.0

        X.append(inp)
        Y.append(tar)

    return np.array(X), np.array(Y)


# 🔥 SHARPER LOSS FUNCTION
def combined_loss(y_true, y_pred):
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return mae + 0.5 * mse


if __name__ == "__main__":
    X, Y = load_data("data/input", "data/target")

    model = build_unet()
    model.compile(optimizer='adam', loss=combined_loss)

    model.fit(X, Y, epochs=80, batch_size=8)

    os.makedirs("models", exist_ok=True)
    model.save("models/unet.h5")

    print("✅ Model trained (sharper)")