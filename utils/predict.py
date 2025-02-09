from tensorflow.keras.applications import Xception
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

import utils.preprocessing as pp

import numpy as np

model = load_model('models/model.h5')

def predict_image(image):
    
    features = pp.xception_bg(image)
    
    print(f"===================>{features.shape}")
    
    prediction = model.predict(features)
    return prediction[0][0]