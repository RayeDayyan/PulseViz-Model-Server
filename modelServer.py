from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import cv2
app = Flask(__name__)

# Load the model (make sure to replace the path with your model's path)
model = tf.keras.models.load_model(r"D:\Study Material\SEMESTER-VII\FYP\Model Server\pulsevizmodel20.h5")


def preprocess_image(image):
    # Resize the image to the model's expected input size
    image = np.array(image)

    image = cv2.resize(image, (720, 720))  # Adjust size as per your model
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Expand dimensions to add the channel axis (grayscale channel)
    image = image[..., np.newaxis]
    # Normalize the image
    image = image / 255.0
    # Expand dimensions to match batch size
    return np.expand_dims(image, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the POST request
    image_file = request.files.get('image')

    if image_file:
        # Open the image
        image = Image.open(io.BytesIO(image_file.read()))

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make a prediction
        prediction = model.predict(processed_image)

        # Assuming the output is a single value between 0 and 1
        result = prediction[0][0] > 0.5  # True if prediction > 0.5, False otherwise

        # Return the result
        return jsonify({'result': bool(result)})


    return jsonify({'error': 'No image provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
