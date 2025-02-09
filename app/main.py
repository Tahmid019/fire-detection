from flask import Flask, request, jsonify
from PIL import Image
from utils.predict import predict_image

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    
    file = request.files['image']
    image = Image.open(file)
    
    prediction = predict_image(image)
    return jsonify({'prediction': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
    
    