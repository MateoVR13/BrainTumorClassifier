# 🔍 BLOQUE: Predicción individual con etiqueta real desde carpeta (sin mostrar confianza)

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Ruta al modelo .h5
model_path = 'models/best_model_brain_tumor.h5'
model = tf.keras.models.load_model(model_path)

# Clases en orden
class_names = ['Healthy', 'Tumor']

# Función para preprocesar imagen
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return tf.expand_dims(img_array, axis=0)

# 🔹 Predicción individual con visualización y etiqueta real
def predict_and_plot(img_path):
    # Obtener la etiqueta real desde el nombre de la carpeta
    true_label = os.path.basename(os.path.dirname(img_path))

    # Preprocesar imagen
    img_tensor = preprocess_image(img_path)

    # Hacer predicción
    pred_prob = model.predict(img_tensor)[0][0]
    pred_label_idx = int(pred_prob > 0.5)
    pred_label = class_names[pred_label_idx]

    # Mostrar resultados sin confianza
    plt.imshow(tf.squeeze(img_tensor))
    plt.title(f"Real: {true_label} | Predicted: {pred_label}")
    plt.axis('off')
    plt.show()

# ✅ USO:
predict_and_plot('test_images/Tumor/Cancer (1196).jpg')