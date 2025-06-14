# 🧠 Brain Tumor Detector

This web application detects brain tumors in MRI images using a deep learning model built with TensorFlow and deployed with Streamlit.

## 🚀 What does this app do?

The model receives a brain MRI image and predicts whether it corresponds to a **healthy brain** or one with a **tumor**. The classification is displayed with a confidence percentage, and the interface allows users to upload images and see results interactively.

## 🧠 How does it work?

This system uses **Transfer Learning** based on the **EfficientNetB0** architecture, a state-of-the-art convolutional neural network pre-trained on the ImageNet dataset. Only the final layers were retrained with MRI images to adapt the model to this specific classification task.

### Prediction process:

1. The uploaded image is resized to 224x224 pixels and normalized.
2. The model makes a binary prediction (0 = healthy, 1 = tumor).
3. The result is interpreted and shown to the user along with the confidence level.

## 📦 Requirements

- Python 3.8 or higher
- TensorFlow
- Streamlit
- Pillow
- NumPy

Install dependencies with:

```bash
pip install -r requirements.txt
