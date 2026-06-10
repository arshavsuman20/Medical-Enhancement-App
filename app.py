import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from streamlit_image_comparison import image_comparison
import time
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

def generate_pdf_report(
    scan_type,
    width,
    height,
    file_size_kb,
    sharpness,
    contrast,
    noise,
    enhancement_strength,
    processing_time
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph(
            "Medical Scan Report",
            styles["Title"]
        )
    )

    elements.append(Spacer(1, 12))

    elements.append(
        Paragraph(
            f"Scan Type: {scan_type}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Resolution: {width} x {height}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"File Size: {file_size_kb:.1f} KB",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Sharpness: {sharpness:.1f}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Contrast: {contrast:.1f}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Noise: {noise:.1f}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Enhancement Strength: {enhancement_strength}%",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Processing Time: {processing_time:.2f} sec",
            styles["BodyText"]
        )
    )

    doc.build(elements)

    pdf = buffer.getvalue()

    buffer.close()

    return pdf


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
# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

st.divider()

st.subheader("Session Analytics")

total_images = len(st.session_state.history)

if total_images > 0:

    avg_impact = sum(
        item["impact"]
        for item in st.session_state.history
    ) / total_images

    st.metric(
        "Images Processed",
        total_images
    )

    st.metric(
        "Average Impact",
        f"{avg_impact:.1f}"
    )

else:

    st.metric(
        "Images Processed",
        0
    )

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

    st.write("Version: 3.0")
    
    st.divider()

    st.subheader("Enhancement History")

    if len(st.session_state.history) == 0:
        st.caption("No images processed yet")

    else:
        for item in reversed(st.session_state.history):
            st.write(
                f"{item['name']} | "
                f"{item['scan_type']} | "
                f"Impact {item['impact']:.1f}"
            )

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

    # -----------------------------------
    # DIFFERENCE MAP
    # -----------------------------------

    difference = cv2.absdiff(
        original,
        result
    )

    difference_gray = cv2.cvtColor(
        difference,
        cv2.COLOR_RGB2GRAY
    )

    difference_heatmap = cv2.applyColorMap(
        difference_gray,
        cv2.COLORMAP_JET
    )

    impact_score = np.mean(
        difference_gray
    )

    difference_gray = cv2.cvtColor(
        difference,
        cv2.COLOR_RGB2GRAY
    )

    impact_score = np.mean(
        difference_gray
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


    st.session_state.history.append(
        {
            "name": uploaded_file.name,
            "scan_type": scan_type,
            "impact": impact_score
        }
    )
    
    st.success(
        f"Processing completed in {processing_time:.2f} seconds"
    )
    st.divider()

    st.subheader("Medical Scan Report")

    r1, r2 = st.columns(2)

    with r1:

        st.write(f"**Scan Type:** {scan_type}")
        st.write(f"**Resolution:** {width} × {height}")
        st.write(f"**File Size:** {file_size_kb:.1f} KB")

    # -----------------------------------
    # QUALITY GRADES
    # -----------------------------------

    if sharpness > 1000:
        sharpness_grade = "Excellent"

    elif sharpness > 300:
        sharpness_grade = "Good"

    else:
        sharpness_grade = "Poor"


    if contrast > 50:
        contrast_grade = "High"

    elif contrast > 30:
        contrast_grade = "Moderate"

    else:
        contrast_grade = "Low"


    if noise < 5:
        noise_grade = "Low"

    elif noise < 15:
        noise_grade = "Moderate"

    else:
        noise_grade = "High"


    with r2:

        st.write(
            f"**Sharpness:** {sharpness_grade} ({sharpness:.1f})"
        )

        st.write(
            f"**Contrast:** {contrast_grade} ({contrast:.1f})"
        )

        st.write(
            f"**Noise:** {noise_grade} ({noise:.1f})"
        )
    st.write(
        f"**Enhancement Strength:** {enhancement_strength}%"
    )

    st.write(
        f"**Processing Time:** {processing_time:.2f} sec"
    )

    st.success(
        "Enhancement Applied Successfully"
    )
    st.divider()

    st.subheader("AI Recommendation")

    if noise > 15:

        recommendation = """
        High noise detected.
        Enhancement strongly recommended.
        """

    elif contrast < 40:

        recommendation = """
        Contrast is below ideal levels.
        Enhancement recommended.
        """

    elif sharpness < 300:

        recommendation = """
        Image appears slightly blurred.
        Sharpness enhancement recommended.
        """

    else:

        recommendation = """
        Image quality is acceptable.
        Enhancement produced minor improvements.
        """

    st.info(recommendation)
    # --------------------------------------------------
    # ENHANCEMENT SUMMARYst.divider()

    st.subheader("Enhancement Summary")

    st.success(
        f"""
    Scan Type: {scan_type}

    Impact Score: {impact_score:.2f}

    Enhancement Strength: {enhancement_strength}%

    Sharpness: {sharpness_grade}

    Contrast: {contrast_grade}

    Noise: {noise_grade}
    """
    )
    st.divider()

    st.subheader("Difference Analysis")

    d1, d2 = st.columns(2)

    with d1:

        st.metric(
            "Impact Score",
            f"{impact_score:.2f}"
        )

    with d2:

        if impact_score < 5:

            st.success(
                "Minor Enhancement Applied"
            )

        elif impact_score < 15:

            st.info(
                "Moderate Enhancement Applied"
            )

        else:

            st.warning(
                "Strong Enhancement Applied"
            )
        
    pdf_report = generate_pdf_report(
        scan_type,
        width,
        height,
        file_size_kb,
        sharpness,
        contrast,
        noise,
        enhancement_strength,
        processing_time
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

    st.subheader("Difference Map")

    overlay = cv2.addWeighted(
        original,
        0.95,
        difference_heatmap,
        0.05,
        0
    )

    st.image(
        overlay,
        caption="Enhancement Impact Overlay",
        use_container_width=True
    )
    
    # --------------------------------------------------
    # DOWNLOADS
    # --------------------------------------------------

    st.divider()

    st.subheader("Downloads")

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(
            label="Download Enhanced Image",
            data=cv2.imencode(
                ".png",
                result
            )[1].tobytes(),
            file_name="enhanced_image.png",
            mime="image/png"
        )

    with col2:

        st.download_button(
            label="Download Medical Report PDF",
            data=pdf_report,
            file_name="medical_scan_report.pdf",
            mime="application/pdf"
        )