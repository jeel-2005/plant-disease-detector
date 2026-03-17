import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os
import tflite_runtime.interpreter as tflite

# -----------------------------
# 🎨 PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")

# -----------------------------
# 🌿 BEAUTIFUL UI
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
}
.stButton>button:hover {
    background-color: #00cc99;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🌱 AI Plant Disease Detection</h1>", unsafe_allow_html=True)

# -----------------------------
# 📥 MODEL DOWNLOAD
# -----------------------------
MODEL_URL = "https://drive.google.com/uc?id=1nIeXE0nB_wIdh2r3e7pxieffQXdEz-aK"
MODEL_PATH = "model.tflite"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading AI model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# -----------------------------
# 🤖 LOAD MODEL
# -----------------------------
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
# 🌿 TREATMENT
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
# 📤 UPLOAD IMAGE
# -----------------------------
uploaded_file = st.file_uploader("📸 Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Prediction
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    # Output
    st.success(f"🌿 Prediction: {predicted_class}")
    st.info(f"📊 Confidence: {confidence:.2f}%")
    st.warning(f"💊 Treatment: {treatment[predicted_class]}")