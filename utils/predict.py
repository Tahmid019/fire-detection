from tensorflow.keras.applications import Xception
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from PIL import Image
import numpy as np

xception_model = Xception(weights='imagenet', include_top=False, pooling='avg')
custom_model = load_model('models/custom_model.h5')

def predict_image(image):
    
    image = image.resize((255,255))
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    
    features = xception_model.predict(image)
    
    prediction = custom_model.predict(features)
    return prediction[0][0]