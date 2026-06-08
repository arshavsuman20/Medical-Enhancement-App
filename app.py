import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from streamlit_image_comparison import image_comparison
import time

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Medical Image Enhancement Platform",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Medical Image Enhancement Platform")

st.markdown("""
Upload a medical image and enhance it using a deep learning model.

Supported formats:

- PNG
- JPG
- JPEG

After enhancement you can compare and download the result.
""")

with st.sidebar:

    st.header("About")

    st.write("""
    Deep learning based medical image enhancement platform.

    Model:
    U-Net Generator

    Input Size:
    128 × 128

    Supported Modalities:
    - MRI
    - CT
    - X-ray
    """)

    st.divider()

    st.write("Version: 1.0")
    
# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_my_model():
    model = tf.saved_model.load("models/final_model")
    return model

model = load_my_model()

# --------------------------------------------------
# IMAGE PROCESSING
# --------------------------------------------------

def process_image(image, model):

    img = np.array(image)

    # Grayscale → RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # RGBA → RGB
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.resize(img, (128, 128))
    img = img / 255.0

    img = np.expand_dims(img, axis=0).astype("float32")

    infer = model.signatures["serving_default"]

    output = infer(tf.constant(img))
    output = list(output.values())[0].numpy()

    output = output[0]
    output = (output * 255).astype("uint8")

    return output

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Medical Image",
    type=["png", "jpg", "jpeg"]
)

# --------------------------------------------------
# INFERENCE
# --------------------------------------------------

if uploaded_file:

    image = Image.open(uploaded_file)

    file_size_kb = uploaded_file.size / 1024

    width, height = image.size

    st.subheader("Image Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Filename", uploaded_file.name)

    with col2:
        st.metric("Resolution", f"{width} × {height}")

    with col3:
        st.metric("File Size", f"{file_size_kb:.1f} KB")

    start_time = time.time()

    with st.spinner("Enhancing image..."):
        result = process_image(image, model)

    processing_time = time.time() - start_time

    st.info(
        f"Processing completed in {processing_time:.2f} seconds"
    )

    st.subheader("Before vs After Comparison")

    try:
        image_comparison(
            img1=image,
            img2=result,
            label1="Original",
            label2="Enhanced",
        )

    except Exception:

        st.warning(
            "Comparison slider unavailable. Showing side-by-side images instead."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original")

        with col2:
            st.image(result, caption="Enhanced")

    st.divider()

    st.subheader("Download Result")

    st.download_button(
        label="Download Enhanced Image",
        data=cv2.imencode(".png", result)[1].tobytes(),
        file_name="enhanced_image.png",
        mime="image/png"
    )