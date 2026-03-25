# Medical Image Enhancement using GAN

A deep learning-based web application that enhances noisy medical images using a Generative Adversarial Network (GAN). The app allows users to upload an image and get an improved version in real time.

---

## Live Demo

https://medical-enhancement-app-vadvkn9rrneucqypm2azwm.streamlit.app/

---

## Features

* Upload medical images (PNG, JPG, JPEG)
* GAN-based image enhancement
* Real-time inference using trained model
* Side-by-side comparison (Original vs Enhanced)
* Download enhanced image

---

## Model Details

* Architecture: GAN / U-Net based generator
* Input size: 128 × 128 images
* Preprocessing:

  * Resize
  * Normalization (0–1)
  * Channel adjustment (RGB)
* Output: Enhanced image with reduced noise

---

## Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* OpenCV
* Streamlit

---

## How to Run Locally

```bash
git clone https://github.com/arshavsuman20/Medical-Enhancement-App.git
cd Medical-Enhancement-App
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
Medical-Enhancement-App/
│
├── app.py                 # Streamlit application
├── requirements.txt       # Dependencies
├── models/
│   └── final_model/       # SavedModel format
├── utils.py (optional)    # Helper functions
├── data/                  # Sample images
└── README.md
```

---

## Challenges & Learnings

* Handled TensorFlow model compatibility issues (.h5 vs SavedModel)
* Resolved deployment errors on Streamlit Cloud
* Fixed OpenCV dependency issues in cloud environments
* Built complete ML inference pipeline from scratch

---

## Use Cases

* Medical image preprocessing
* Noise reduction in scans (MRI/CT/X-ray)
* Educational demo for GAN-based enhancement

---

## Future Improvements

* Train on real MRI/CT datasets
* Improve model architecture (Pix2Pix / U-Net GAN)
* Add interactive comparison slider
* Enhance UI/UX

---

## Author

**Arshav Suman**
GitHub: https://github.com/arshavsuman20

---

## If you found this useful, consider giving it a star!
