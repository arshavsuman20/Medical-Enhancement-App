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

The platform will:
1. Analyze image quality
2. Recommend enhancement
3. Enhance image using AI
4. Allow comparison and download
""")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("About")

    st.write("""
    Deep Learning Based Medical Image Enhancement Platform

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

    st.write("Version: 2.1")

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_my_model():
    model = tf.saved_model.load("models/final_model")
    return model

model = load_my_model()

# --------------------------------------------------
# IMAGE QUALITY ASSESSMENT
# --------------------------------------------------

def assess_image_quality(image):

    img = np.array(image)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    contrast = gray.std()

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    noise = np.mean(
        cv2.absdiff(gray, blurred)
    )

    return sharpness, contrast, noise

# --------------------------------------------------
# IMAGE PROCESSING
# --------------------------------------------------

def process_image(image, model):

    img = np.array(image)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.resize(img, (128, 128))
    img = img / 255.0

    img = np.expand_dims(
        img,
        axis=0
    ).astype("float32")

    infer = model.signatures["serving_default"]

    output = infer(tf.constant(img))

    output = list(output.values())[0].numpy()

    output = output[0]

    output = (
        output * 255
    ).astype("uint8")

    return output

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Medical Image",
    type=["png", "jpg", "jpeg"]
)
scan_type = st.selectbox(
    "Select Scan Type",
    [
        "MRI",
        "CT",
        "X-Ray",
        "Ultrasound",
        "Retinal",
        "Other"
    ]
)
# --------------------------------------------------
# MAIN APP
# --------------------------------------------------

if uploaded_file:

    image = Image.open(uploaded_file)

    sharpness, contrast, noise = assess_image_quality(image)

    file_size_kb = uploaded_file.size / 1024

    width, height = image.size

    # --------------------------------------------------
    # IMAGE INFO
    # --------------------------------------------------

    st.subheader("Image Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Filename",
            uploaded_file.name
        )

    with col2:
        st.metric(
            "Resolution",
            f"{width} × {height}"
        )

    with col3:
        st.metric(
            "File Size",
            f"{file_size_kb:.1f} KB"
        )

    with col4:
        st.metric(
            "Scan Type",
            scan_type
        )
    # --------------------------------------------------
    # QUALITY DASHBOARD
    # --------------------------------------------------

    st.divider()

    st.subheader("AI Quality Assessment")

    q1, q2, q3 = st.columns(3)

    with q1:
        st.metric(
            "Sharpness",
            f"{sharpness:.1f}"
        )

    with q2:
        st.metric(
            "Contrast",
            f"{contrast:.1f}"
        )

    with q3:
        st.metric(
            "Noise",
            f"{noise:.1f}"
        )

    # --------------------------------------------------
    # RECOMMENDATION ENGINE
    # --------------------------------------------------

    st.subheader("Quality Analysis")

    issues = []

    if sharpness < 100:
        issues.append("Low Sharpness")

    if contrast < 40:
        issues.append("Low Contrast")

    if noise > 15:
        issues.append("High Noise")

    if len(issues) == 0:

        st.success(
            "Image quality looks good. Enhancement may provide minor improvements."
        )

    else:

        for issue in issues:
            st.warning(issue)

        st.info(
            "Enhancement Recommended"
        )

    # --------------------------------------------------
    # ENHANCEMENT
    # --------------------------------------------------

    st.subheader("Enhancement Settings")

    enhancement_strength = st.slider(
        "Enhancement Strength",
        min_value=0,
        max_value=100,
        value=100,
        step=5
    )
    start_time = time.time()

    with st.spinner(
        "Enhancing image..."
    ):
        result = process_image(
            image,
            model
        )

    original = np.array(image)

    if len(original.shape) == 2:
        original = cv2.cvtColor(
            original,
            cv2.COLOR_GRAY2RGB
        )

    elif len(original.shape) == 3 and original.shape[2] == 4:
        original = cv2.cvtColor(
            original,
            cv2.COLOR_RGBA2RGB
        )

    original = cv2.resize(
        original,
        (128, 128)
    )

    alpha = enhancement_strength / 100.0

    result = cv2.addWeighted(
        result,
        alpha,
        original,
        1 - alpha,
        0
    )
    processing_time = (
        time.time()
        - start_time
    )

    st.success(
        f"Processing completed in {processing_time:.2f} seconds"
    )

    # --------------------------------------------------
    # COMPARISON
    # --------------------------------------------------

    st.divider()

    st.subheader(
        "Before vs After Comparison"
    )

    try:

        image_comparison(
            img1=image,
            img2=result,
            label1="Original",
            label2="Enhanced",
        )

    except Exception:

        st.warning(
            "Comparison slider unavailable. Showing side-by-side images."
        )

        c1, c2 = st.columns(2)

        with c1:
            st.image(
                image,
                caption="Original"
            )

        with c2:
            st.image(
                result,
                caption="Enhanced"
            )

    # --------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------

    st.divider()

    st.subheader(
        "Download Result"
    )

    st.download_button(
        label="Download Enhanced Image",
        data=cv2.imencode(
            ".png",
            result
        )[1].tobytes(),
        file_name="enhanced_image.png",
        mime="image/png"
    )