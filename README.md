# Medical Image Enhancement Platform

A deep learning-powered web application for enhancing medical images using a U-Net-based image restoration model. Users can upload medical images, visualize improvements with an interactive comparison slider, and download the enhanced results instantly.

---

## Live Demo

https://medical-enhancement-app-vadvkn9rrneucqypm2azwm.streamlit.app/

---

## Features

### Medical Image Enhancement

* Deep learning-based image enhancement
* Noise reduction and image refinement
* Real-time inference using TensorFlow

### Interactive Comparison Slider

* Drag-and-drop comparison between original and enhanced images
* Side-by-side visual evaluation
* Professional image review experience

### Smart Upload Pipeline

* Supports PNG, JPG, and JPEG images
* Handles RGB, RGBA, and grayscale inputs
* Automatic preprocessing and resizing

### Download Enhanced Results

* Download enhanced images instantly
* PNG output format

### Image Analytics

* Original image resolution
* File size information
* Processing time tracking

---

## Technology Stack

### Frontend

* Streamlit

### Deep Learning

* TensorFlow
* Keras

### Image Processing

* OpenCV
* Pillow
* NumPy

### Visualization

* streamlit-image-comparison

---

## Model Architecture

The application uses a U-Net-inspired encoder-decoder architecture trained for medical image enhancement tasks.

### Input

* Image Size: 128 × 128
* 3-channel RGB input

### Output

* Enhanced image
* Reduced visual noise
* Improved structural clarity

---

## Project Workflow

1. User uploads a medical image.
2. Image is validated and preprocessed.
3. Deep learning model performs enhancement.
4. Enhanced image is generated.
5. Interactive slider displays before/after comparison.
6. User downloads the enhanced result.

---

## Local Installation

Clone the repository:

```bash
git clone https://github.com/arshavsuman20/Medical-Enhancement-App.git
cd Medical-Enhancement-App
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Project Structure

```text
Medical-Enhancement-App/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── final_model/
│
├── data/
│   ├── input/
│   ├── target/
│   ├── clean_images/
│   └── clean_images_processed/
│
├── train.py
├── create_dataset.py
├── prepare_mri_dataset.py
├── evaluate.py
└── convert_model.py
```

---

## Key Highlights

* End-to-end ML deployment project
* Publicly accessible web application
* Real-time deep learning inference
* Interactive image comparison slider
* Medical imaging use case
* Production-ready deployment on Streamlit Cloud

---

## Future Improvements

### Version 2

* Enhanced GAN architecture
* Multi-modal medical image support
* Higher-resolution inference
* Batch image processing

### Version 3

* User authentication
* Cloud storage integration
* Patient report generation
* AI-assisted image quality assessment

---

## Author

Arshav Suman

GitHub:
https://github.com/arshavsuman20

---

## License

This project is intended for educational and research purposes.
