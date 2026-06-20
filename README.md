---
title: Diabetic Retinopathy Detection System
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
---

# 👁️ Diabetic Retinopathy Detection System

An AI-powered diagnostic screening tool designed to detect the severity stages of **Diabetic Retinopathy (DR)** from retinal fundus photographs. 

This project was built for a college engineering presentation, combining image preprocessing, generative model balancing (DDPM), and a deep convolutional neural network (ResNet18).

---

## 🚀 Live Deployment
The application is deployed online and active 24/7 on Hugging Face Spaces. You can try the system live here:

👉 **[Live Demo: Diabetic Retinopathy Detection System](https://huggingface.co/spaces/mohammedathiqsyed/diabetic-retinopathy-detection)**

---

## 🛠️ How It Works

1. **Retinal Preprocessing (Green Channel + CLAHE):**
   * The uploaded color fundus image is converted, and the **green channel** is extracted (where diabetic lesions, microaneurysms, and hemorrhages are most prominent).
   * **CLAHE (Contrast Limited Adaptive Histogram Equalization)** is applied to enhance details and contrast.
   * A **Gaussian Blur** is used to smooth out high-frequency noise.
2. **Deep Learning Inference:**
   * The preprocessed image is fed into a **ResNet18** model trained on the Diabetic Retinopathy dataset.
   * The model outputs classification probabilities across 5 stages of severity.
3. **Clinical Recommendations:**
   * The system displays the predicted DR stage, classification confidence bars, and a list of recommended medical precautions.

---

## 📊 Model & Features

* **Architecture:** ResNet18 (pretrained on ImageNet, fine-tuned).
* **Imbalance Resolution:** Handled in-memory during training using a PyTorch `WeightedRandomSampler` to ensure balanced class learning.
* **Output Classes:**
  * `Healthy`
  * `Mild DR`
  * `Moderate DR`
  * `Severe DR`
  * `Proliferate DR`
