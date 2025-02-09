from tensorflow.keras.applications import Xception
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

import utils.preprocessing as pp
import cv2

import numpy as np

model = load_model('models/model2.h5')

def predict_image(imagep):
    
    features = pp.xception_bg(imagep)
    print(f" === features shape: { features.shape }")
    
    prediction = np.argmax(model.predict(features), axis=1)
    return prediction