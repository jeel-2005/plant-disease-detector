import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os

# Hide TensorFlow logs (safe even if TF not installed)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Use keras instead of tensorflow.keras
from keras.models import load_model

# -----------------------------
# 🎨 PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")

# -----------------------------
# 🌿 CUSTOM UI (ANIMATED)
# -----------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
.main {
    background-color: rgba(0,0,0,0.6);
    border-radius: 20px;
    padding: 20px;
}
h1 {
    text-align: center;
    color: #00ffcc;
    animation: fadeIn 2s ease-in-out;
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
.stButton>button {
    background-color: #00ffcc;
    color: black;
    border-radius: 10px;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #00cc99;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🌱 AI Plant Disease Detection</h1>", unsafe_allow_html=True)

# -----------------------------
# 📥 DOWNLOAD MODEL FROM DRIVE
# -----------------------------
MODEL_URL = "https://drive.google.com/uc?id=1i8xdXb0nR1NPTguAYQTDfG9zsPSLbZqd"
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading AI model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

# -----------------------------
# 📊 CLASS NAMES (16)
# -----------------------------
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
    "Tomato Leaf Curl"
]

# -----------------------------
# 🌿 TREATMENT SUGGESTIONS
# -----------------------------
treatment = {
    "Pepper Bell Bacterial Spot": "Use copper-based fungicide.",
    "Pepper Bell Healthy": "No treatment needed.",
    "Potato Early Blight": "Apply fungicide and remove infected leaves.",
    "Potato Healthy": "Healthy plant.",
    "Potato Late Blight": "Use resistant varieties and fungicide.",
    "Tomato Target Spot": "Use fungicide and proper spacing.",
    "Tomato Mosaic Virus": "Remove infected plants.",
    "Tomato Yellow Leaf Curl Virus": "Control whiteflies.",
    "Tomato Bacterial Spot": "Use copper spray.",
    "Tomato Early Blight": "Use fungicide.",
    "Tomato Healthy": "Healthy plant.",
    "Tomato Late Blight": "Apply fungicide immediately.",
    "Tomato Leaf Mold": "Improve air circulation.",
    "Tomato Septoria Leaf Spot": "Remove infected leaves.",
    "Tomato Spider Mites": "Use neem oil spray.",
    "Tomato Leaf Curl": "Control insects."
}

# -----------------------------
# 📤 IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📸 Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    # Output
    st.success(f"🌿 Prediction: {predicted_class}")
    st.info(f"📊 Confidence: {confidence:.2f}%")

    # Treatment
    st.warning(f"💊 Treatment: {treatment[predicted_class]}")