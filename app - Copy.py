from flask import Flask, request, jsonify
import cv2
import numpy as np
from model import train_models, predict_cancer
from flask_cors import CORS  
import pickle

app = Flask(__name__)


CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}}) 


svm_model, pca, svm_acc, kmeans_acc = train_models()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        
        img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

      
        prediction = predict_cancer(img, svm_model, pca)

        return jsonify({
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
