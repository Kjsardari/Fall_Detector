from flask import Flask, render_template, request
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
model = tf.keras.models.load_model('model/fall_detector.h5')
IMG_SIZE = (128, 128)  # Must match training

# Predict function
def predict_fall(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize like training
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)

    # Make prediction
    prediction = model.predict(img_array)[0][0]
    print(f"Confidence: {prediction:.4f}")  # Debugging info

    # Return result with confidence
    if prediction > 0.5:
        return f"✅ Normal (Confidence: {prediction:.2f})"
    else:
        return f"🚨 Fall Detected (Confidence: {1 - prediction:.2f})"

# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            result = predict_fall(filepath)
            return render_template('index.html', result=result, img_path=filepath)
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
