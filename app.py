import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os

# ================= DOWNLOAD MODEL =================
MODEL_URL = "https://drive.google.com/uc?id=1i8xdXb0nR1NPTguAYQTDfG9zsPSLbZqd"
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("🌿 Downloading AI Model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ================= LOAD MODEL =================
model = tf.keras.models.load_model(MODEL_PATH)

# ================= CLASS NAMES =================
class_names = [
    "Pepper Bell Bacterial Spot",
    "Pepper Bell Healthy",
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight",
    "Tomato Target Spot",
    "Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Healthy",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Unknown Disease"
]

# ================= TREATMENT =================
treatments = {
    "Pepper Bell Bacterial Spot": "Use copper-based fungicides.",
    "Pepper Bell Healthy": "No treatment needed.",
    "Potato Early Blight": "Use fungicide and remove infected leaves.",
    "Potato Healthy": "Healthy plant.",
    "Potato Late Blight": "Apply fungicide immediately.",
    "Tomato Target Spot": "Use resistant varieties.",
    "Tomato Mosaic Virus": "Remove infected plants.",
    "Tomato Yellow Leaf Curl Virus": "Control whiteflies.",
    "Tomato Bacterial Spot": "Use copper sprays.",
    "Tomato Early Blight": "Apply fungicide.",
    "Tomato Healthy": "Healthy plant.",
    "Tomato Late Blight": "Use fungicides urgently.",
    "Tomato Leaf Mold": "Improve air circulation.",
    "Tomato Septoria Leaf Spot": "Remove infected leaves.",
    "Tomato Spider Mites": "Use insecticidal soap.",
    "Unknown Disease": "Consult expert."
}

# ================= UI =================
st.set_page_config(page_title="🌿 AI Plant Doctor", layout="wide")

# 🌈 Animated Background + Glass UI
st.markdown("""
<style>
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.stApp {
    background: linear-gradient(-45deg, #43cea2, #185a9d, #56ab2f, #a8e063);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
}

/* Glass card */
.card {
    background: rgba(255, 255, 255, 0.15);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    margin-top: 20px;
}

/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
}

/* Upload box */
.stFileUploader {
    border-radius: 15px;
}

/* Result boxes */
.result {
    font-size: 22px;
    font-weight: bold;
    color: white;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌱 AI Plant Disease Detector</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📸 Upload Leaf Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Analyzing Plant..."):
        prediction = model.predict(img_array)

    pred_len = prediction.shape[1]
    class_len = len(class_names)

    if pred_len != class_len:
        st.error(f"⚠ Model Output: {pred_len}, Classes: {class_len}")
    else:
        index = np.argmax(prediction)
        confidence = prediction[0][index] * 100
        disease = class_names[index]

        st.markdown(f'<div class="result">🌿 Disease: {disease}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result">📊 Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result">💊 Treatment: {treatments[disease]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)