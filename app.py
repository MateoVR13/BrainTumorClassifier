import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Detector de Tumores Cerebrales",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CACHING DEL MODELO ---
@st.cache_resource
def load_keras_model():
    """Carga el modelo H5 pre-entrenado."""
    try:
        model = tf.keras.models.load_model('brain_tumor_classifier.h5')
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.stop()

# --- PREPROCESAMIENTO DE LA IMAGEN ---
def preprocess_image(image):
    """
    Preprocesa la imagen para que coincida con el formato de entrada del modelo.
    """
    image = image.resize((224, 224))
    img_array = np.array(image)
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    processed_image = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return processed_image

# --- FUNCIÓN PRINCIPAL DE LA APP ---
def main():
    model = load_keras_model()

    # --- INTERFAZ DE USUARIO CON STREAMLIT ---
    
    st.title("🧠 Detector de Tumores Cerebrales")
    st.caption("Una aplicación para clasificar imágenes de resonancia magnética cerebral.")

    st.markdown("""
    Sube una imagen de una resonancia magnética del cerebro y el modelo predecirá si la imagen
    muestra un cerebro sano o uno con un tumor.
    """)

    # --- CARGA DE ARCHIVOS ---
    uploaded_file = st.file_uploader(
        "Elige una imagen...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Imagen subida', use_container_width=True)

        with st.spinner('Clasificando...'):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
        
        score = prediction[0][0]
        
        with col2:
            st.subheader("Resultado del Diagnóstico:")
            
            if score > 0.5:
                confidence = score * 100
                st.error(f"**Diagnóstico: Tumor Detectado**")
                st.info(f"**Confianza:** {confidence:.2f}%")
            else:
                confidence = (1 - score) * 100
                st.success(f"**Diagnóstico: Cerebro Sano**")
                st.info(f"**Confianza:** {confidence:.2f}%")

            st.warning("""
            **Descargo de responsabilidad:** Este modelo es una herramienta de apoyo y no reemplaza 
            el diagnóstico de un profesional médico cualificado.
            """)

    # --- BARRA LATERAL CON INFORMACIÓN ---
    st.sidebar.title("Sobre el Proyecto")
    st.sidebar.info(
        "Esta aplicación utiliza un modelo de Red Neuronal Convolucional (CNN) para la clasificación de imágenes médicas."
    )
    
    # --- NUEVA SECCIÓN: INFORMACIÓN TEÓRICA ---
    st.sidebar.header("¿Cómo funciona el modelo?")
    st.sidebar.markdown(
        """
        El modelo se basa en una técnica llamada **Aprendizaje por Transferencia** (*Transfer Learning*):

        1.  **Modelo Base (EfficientNetB0):** En lugar de construir una red desde cero, utilizamos 'EfficientNetB0', un modelo de última generación ya pre-entrenado en millones de imágenes del dataset *ImageNet*. Este modelo ya es un experto en reconocer patrones, texturas y formas complejas.

        2.  **Congelación de Capas:** "Congelamos" las capas del modelo base para que no pierdan su conocimiento general sobre cómo "ver" las imágenes.

        3.  **Capa de Clasificación:** Añadimos nuevas capas personalizadas al final del modelo. Solo estas nuevas capas se entrenan con nuestro dataset de resonancias. De esta forma, el modelo aprende a usar el conocimiento de EfficientNetB0 para la tarea específica de distinguir entre un cerebro sano y uno con tumor.

        4.  **Salida Sigmoid:** La capa final utiliza una función de activación 'sigmoid', que produce un único valor entre 0 y 1. Este valor se interpreta como la probabilidad de que la imagen contenga un tumor, permitiendo una clasificación binaria.
        """
    )
    
    # --- NUEVA SECCIÓN: CRÉDITOS ---
    st.sidebar.header("Presentado por:")
    st.sidebar.markdown("""
    - Isabella Yusunguaira
    - Mariana Neira
    - Maria Macías
    """)


if __name__ == '__main__':
    main()