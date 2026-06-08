# Medical Image Enhancement Platform

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![License](https://img.shields.io/badge/License-Educational-green)
A deep learning-powered web application that enhances medical images using a U-Net-based image restoration model. The platform allows users to upload images, compare original and enhanced outputs using an interactive slider, and download the processed result in real time.

---

## Live Demo

[![Live Demo](https://img.shields.io/badge/Live-Demo-success?style=for-the-badge)](https://medical-enhancement-app-vadvkn9rrneucqypm2azwm.streamlit.app/)
---

## Features

### Deep Learning-Based Enhancement

* Medical image enhancement using TensorFlow
* U-Net-inspired architecture
* Real-time image processing and inference

### Interactive Comparison Slider

* Drag-and-drop before/after comparison
* Instant visual assessment of enhancement quality
* User-friendly image review experience

### Image Upload Pipeline

* Supports PNG, JPG, and JPEG formats
* Handles RGB, RGBA, and grayscale images
* Automatic preprocessing and normalization

### Image Analytics

* Displays uploaded filename
* Shows image resolution
* Displays file size information
* Tracks processing time

### Download Support

* Download enhanced images directly
* PNG output generation

### Deployment

* Publicly accessible web application
* Hosted on Streamlit Cloud
* Integrated with GitHub for automated deployment

---

## Application Screenshots

### Homepage

![Homepage](assets/screenshots/home_page.png)

---

### Image Metadata & Processing Information

![Metadata](assets/screenshots/upload.png)

---

### Interactive Before vs After Comparison

![Comparison Slider](assets/screenshots/slider.png)
![Downloading](assets/screenshots/download.png)

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

### Deployment

* Streamlit Cloud
* GitHub

---

## Model Architecture

The application uses a U-Net-inspired encoder-decoder architecture trained for medical image enhancement.

### Input

* Image Size: 128 × 128
* RGB image input

### Output

* Enhanced image
* Reduced visual noise
* Improved structural clarity

---

## Workflow

1. Upload a medical image.
2. Image is validated and preprocessed.
3. Deep learning model performs enhancement.
4. Processing statistics are generated.
5. Interactive comparison slider displays results.
6. Enhanced image can be downloaded.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/arshavsuman20/Medical-Enhancement-App.git
cd Medical-Enhancement-App
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
streamlit run app.py
```

---

## Project Structure

```text
Medical-Enhancement-App/
│
├── assets/
│   └── screenshots/
│       ├── homepage.png
│       ├── metadata.png
│       └── comparison-slider.png
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── final_model/
│
├── data/
│
├── train.py
├── create_dataset.py
├── prepare_mri_dataset.py
├── evaluate.py
└── convert_model.py
```

---

## Key Achievements

* Developed and deployed a complete end-to-end deep learning application.
* Implemented real-time medical image enhancement.
* Integrated an interactive before/after comparison slider.
* Built a robust image upload and preprocessing pipeline.
* Added image metadata extraction and processing-time tracking.
* Deployed the application publicly using Streamlit Cloud.

---

## Future Enhancements

### Version 2

* Improved enhancement model
* Support for additional medical imaging modalities
* Higher-resolution inference
* Batch image processing

### Version 3

* User authentication
* Cloud-based image storage
* AI-generated enhancement reports
* Advanced image quality assessment

---

## Author

**Arshav Suman**

GitHub: https://github.com/arshavsuman20

---

## License

This project is intended for educational, research, and portfolio purposes.
