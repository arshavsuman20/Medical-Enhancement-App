from xml.parsers.expat import model

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
import streamlit as st
st.title("Medical Image Enhancement")

@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("models/final_model")
    return model

model = load_my_model()
def process_image(image, model):
    

    # Convert PIL → numpy
    img = np.array(image)

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize to model input (VERY IMPORTANT)
    img = cv2.resize(img, (128, 128))   # <-- match your model size

    # Normalize
    img = img / 255.0

    # Add channel dimension
    img = np.expand_dims(img, axis=-1)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Predict
    output = model.predict(img)

    # Remove batch + channel dims
    output = output[0, :, :, 0]

    # Convert back to 0–255
    output = (output * 255).astype("uint8")

    return output

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original")

    result = process_image(image, model)

    with col2:
        st.image(result, caption="Enhanced")

    st.download_button(
        "Download",
        data=cv2.imencode(".png", result)[1].tobytes(),
        file_name="enhanced.png"
    )
