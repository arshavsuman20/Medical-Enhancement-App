import tensorflow as tf

# Load old model
model = tf.keras.models.load_model("models/unet.h5", compile=False)

# Export as SavedModel (NEW WAY)
model.export("models/final_model")

print("Model converted successfully!")