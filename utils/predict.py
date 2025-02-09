from tensorflow.keras.applications import Xception
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

import utils.preprocessing as pp

import numpy as np

custom_model = load_model('models/custom_model.h5')

def predict_image(image):
    
    features = pp.xception_bg(image)
    
    prediction = custom_model.predict(features)
    return prediction[0][0]