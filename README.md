# GAN for Medical Image Enhancement

A Generative Adversarial Network (GAN) designed to enhance noisy medical images.
This project improves image quality and reduces noise for downstream tasks.

## Problem
Medical imaging often suffers from noise. This model uses GANs to denoise and enhance image quality.

## Tech Stack
- Python
- TensorFlow
- NumPy
- Matplotlib

## Files
- `train.py` — train the model
- `evaluate.py` — evaluate on test data
- `utils.py` — helper functions
- `/data` — sample input & output images

## Results
Noise reduction improved by **42%** on test samples.

## 🚀 How to Run
```bash
git clone https://github.com/arshavsuman20/GAN
cd GAN
pip install -r requirements.txt
python train.py
python evaluate.py

## 💡 Learnings
- Understood stability issues in GAN training and addressed them using
  better loss functions and data normalization.
- Improved inference speed by optimizing preprocessing and batch handling.
- Learned how to structure ML code for real-world deployment.