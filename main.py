import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Image dimensions
img_rows, img_cols, channels = 28, 28, 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100  # Size of noise vector

def build_generator():
    model = models.Sequential()
    model.add(layers.Dense(256, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(np.prod(img_shape), activation='tanh'))
    model.add(layers.Reshape(img_shape))
    return model

def build_discriminator():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=img_shape))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def compile_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    discriminator.trainable = False

    gan_input = layers.Input(shape=(latent_dim,))
    img = generator(gan_input)
    validity = discriminator(img)

    gan = models.Model(gan_input, validity)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

def train_gan(generator, discriminator, gan, epochs, batch_size=64, sample_interval=100):
    # Load and preprocess the dataset (using MNIST as stand-in for medical images)
    (X_train, _), (_, _) = mnist.load_data()

    # Normalize to [-1,1]
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=-1)  # shape (num_samples, 28, 28, 1)

    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):

        #  Train Discriminator

        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_images = generator.predict(noise, verbose=0)

        d_loss_real = discriminator.train_on_batch(real_images, valid)
        d_loss_fake = discriminator.train_on_batch(generated_images, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        #  Train Generator

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, valid)

        # Print progress
        if epoch % sample_interval == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

if __name__ == '__main__':
    generator = build_generator()
    discriminator = build_discriminator()
    gan = compile_gan(generator, discriminator)
    train_gan(generator, discriminator, gan, epochs=500, batch_size=64, sample_interval=100)

