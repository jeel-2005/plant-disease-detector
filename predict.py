import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model.h5")

def predict_disease(image):

    img = cv2.resize(image,(224,224))
    img = img/255
    img = np.reshape(img,[1,224,224,3])

    prediction = model.predict(img)

    return prediction