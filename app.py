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
    model = tf.saved_model.load("models/final_model")
    return model

model = load_my_model()
def process_image(image, model):
    import numpy as np
    import cv2
    import tensorflow as tf

    img = np.array(image)

    # Ensure 3 channels
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.resize(img, (128, 128))
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0).astype("float32")

    # 🔥 CORRECT WAY FOR SavedModel
    infer = model.signatures["serving_default"]
    output = infer(tf.constant(img))

    # Extract tensor
    output = list(output.values())[0].numpy()

    # Remove batch dimension
    output = output[0]

    # Convert to image
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
