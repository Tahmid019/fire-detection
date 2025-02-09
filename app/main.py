from flask import Flask, request, jsonify
from PIL import Image
from utils.predict import predict_image
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    
    file = request.files['image']
    image = Image.open(file)
    
    prediction = predict_image(image)
    return jsonify({'prediction': prediction.tolist() if isinstance(prediction, np.ndarray) else prediction})

if __name__ == '__main__':
    app.run(debug=True)
    
    