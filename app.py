import streamlit as st
import tensorflow as tf
from PIL import Image
import os

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Detector de Tumores Cerebrales",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- CACHING DEL MODELO ---
# Usamos st.cache_resource para cargar el modelo solo una vez.
@st.cache_resource
def load_model():
    """Carga el modelo Keras desde la ruta especificada."""
    model_path = 'models/best_model_brain_tumor.h5'
    if not os.path.exists(model_path):
        st.error(f"Error: No se encuentra el archivo del modelo en la ruta: {model_path}")
        st.error("Asegúrate de que la carpeta 'models' y el archivo 'best_model_brain_tumor.h5' existan.")
        return None
    try:
        model = tf.keras.models.load_model(model_path, compile=False) # compile=False puede acelerar la carga
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# --- FUNCIONES DE PROCESAMIENTO ---
def preprocess_image(image_data, target_size=(224, 224)):
    """Preprocesa la imagen subida para que sea compatible con el modelo."""
    img = Image.open(image_data)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalizar
    img_array = tf.expand_dims(img_array, axis=0)  # Añadir dimensión de batch
    return img_array

# CAMBIO: La función ahora solo devuelve la etiqueta, sin la confianza.
def make_prediction(model, processed_image):
    """Realiza la predicción y devuelve únicamente la etiqueta predicha."""
    prediction = model.predict(processed_image)[0][0]
    predicted_class_index = int(prediction > 0.5)
    class_names = ['Sano', 'Tumor']
    predicted_label = class_names[predicted_class_index]
    return predicted_label

# --- INTERFAZ DE LA APLICACIÓN ---
st.title("Detector de Tumores Cerebrales con EfficientNetB0")
st.markdown("""
Esta aplicación utiliza un modelo de **Red Neuronal Convolucional (CNN) EfficientNetB0** para clasificar imágenes de resonancia magnética (MRI) del cerebro. 
Sube una imagen para determinar si es clasificada como **"Sana"** o si presenta indicios de un **"Tumor"**.
""")

# Cargar el modelo
model = load_model()

if model:
    # --- BARRA LATERAL PARA LA CARGA DE ARCHIVOS ---
    st.sidebar.header("Carga tu imagen de MRI")
    uploaded_file = st.sidebar.file_uploader(
        "Elige una imagen...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Mostrar la imagen subida
        st.image(uploaded_file, caption="Imagen de MRI subida.", use_container_width =True)
        st.write("")

        # Botón para iniciar la clasificación
        if st.button("Clasificar Imagen", type="primary"):
            with st.spinner('Analizando la imagen... por favor, espera.'):
                # Preprocesar la imagen
                processed_image = preprocess_image(uploaded_file)
                
                # CAMBIO: Se obtiene solo la etiqueta de la predicción.
                predicted_label = make_prediction(model, processed_image)
                
                # Mostrar los resultados
                st.subheader("🔮 Resultado del Análisis")
                
                # CAMBIO: Se muestra el resultado y se añade una advertencia si es "Tumor".
                if predicted_label == 'Tumor':
                    st.error(f"**Diagnóstico Predicho: {predicted_label}**")
                    st.warning(
                        "**⚠️ Advertencia Importante:** Aunque el modelo ha sido entrenado para ser preciso, "
                        "esta herramienta es solo para fines informativos y educativos. "
                        "**Un diagnóstico real debe ser realizado por un profesional de la salud cualificado.** "
                        "No utilice este resultado para tomar decisiones médicas."
                    )
                else:
                    st.success(f"**Diagnóstico Predicho: {predicted_label}**")

else:
    st.warning("El modelo no pudo ser cargado. Por favor, revisa la configuración.")

# --- SECCIÓN DE INFORMACIÓN TEÓRICA ---
st.markdown("---")
with st.expander("ℹ️ ¿Cómo funciona este modelo? - Información Teórica"):
    st.markdown("""
    ### Clasificación Binaria
    El problema que resolvemos aquí es una **clasificación binaria**. Esto significa que solo hay dos posibles resultados o "clases":
    - **Sano:** La imagen no muestra evidencia de un tumor.
    - **Tumor:** La imagen muestra características asociadas a un tumor cerebral.
    El objetivo del modelo es aprender a distinguir entre estas dos clases a partir de los píxeles de una imagen.

    ### Redes Neuronales Convolucionales (CNN)
    Para tareas de visión por computadora como esta, se utilizan **Redes Neuronales Convolucionales (CNNs)**. Son un tipo de algoritmo de Deep Learning especialmente diseñado para procesar datos con una estructura de rejilla, como las imágenes.
    
    Una CNN aprende a identificar patrones y características de forma jerárquica:
    1.  **Capas Convolucionales:** Actúan como filtros que detectan características simples como bordes y texturas.
    2.  **Capas de Agrupación (Pooling):** Reducen el tamaño de la imagen para hacer el proceso más eficiente.
    3.  **Jerarquía de Características:** A medida que la información avanza por la red, las capas más profundas combinan estas características para reconocer patrones más complejos, como las formas de un tejido sano o las anomalías de un tumor.
    4.  **Capas Densas (Clasificador):** Al final, estas capas toman las características aprendidas y toman la decisión final: ¿la imagen corresponde a "Sano" o a "Tumor"?
    
    Este modelo fue entrenado con miles de imágenes previamente etiquetadas para aprender a realizar esta tarea.
    """)

# --- PIE DE PÁGINA ---
st.sidebar.markdown("---")
st.sidebar.info("App desarrollada para demostrar la clasificación de imágenes con TensorFlow y Streamlit.")