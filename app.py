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
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    img = cv2.resize(image_np, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    output = model.predict(img)
    output = (output[0] * 255).astype(np.uint8)

    # resize back
    output = cv2.resize(output, (w, h))

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
