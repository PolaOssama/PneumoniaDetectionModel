from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import requests
import tempfile
import sys
import urllib.request
# Allowed extensions for the uploaded file
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# Initialize the Flask application
app = Flask(__name__, static_url_path='',
            static_folder='static',
            template_folder='templates')
# Path to the model
MODEL_PATH = 'model_vgg16.h5'

# Load your trained model or raise an exception

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile()
    print('.............Model loaded. Start serving..........\n\n\n\n\n')
except OSError as e:
    print(
        f".............Model file not found ..........\n\n\n\n\n : {MODEL_PATH}")
    sys.exit(1)
except Exception as e:
    print(f"............Error loading model ..........\n\n\n\n\n: {e}")
    sys.exit(1)
# Function to predict the class of an image


def model_predict(img_path, model):
    # Target size must agree with what the trained model expects!!
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    # Preprocessing the image
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255
    # Predicting the class
    classes = model.predict(x)
    return classes
 # Route for the POST request


@app.route('/', methods=['POST'])
def upload():
    # Get the file from post request
    if not request.is_json:
        return jsonify({"Error": "Request must be in JSON format"}), 400
     # Get the URL of the image
    url = request.json['url']
    if not url:
        return jsonify({'Error': 'Non-Valid Please Insert Image'})

    if not any(url.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return jsonify({'Error': 'URL has an unsupported image extension'}), 400
    
    else:
        try:
            response = requests.get(url)
            if response.status_code == 404:
              return jsonify({'Error': 'The URL is not valid or could not be found'})
            else:
                image_url = tf.keras.utils.get_file('Court', origin=url)
                img = tf.keras.preprocessing.image.load_img(
                    image_url, target_size=(224, 224))
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    img.save(f, format='JPEG')
                    temp_path = f.name
                    os.remove(image_url)
                # Make prediction
                preds = model_predict(temp_path, model)
                # Prepare the response
                str1 = 'Positive'
                str2 = 'Negative'
                if preds[0][1] > 0.5:
                    return({'The prediction is': f'{str1}'})
                else:
                    return({'The prediction is': f'{str2}'})
        except requests.exceptions.HTTPError as e:
            return jsonify({'Error': f'Response returned {e.response.status_code}'})
        except requests.exceptions.ConnectionError:
            return jsonify({'Error': 'Failed to establish a connection to the server.'})
        except requests.exceptions.Timeout:
            return jsonify({'Error': 'Request timed out.'})
        except requests.exceptions.RequestException:
            return jsonify({'Error': 'An unexpected error occurred.'})
    return jsonify({'Error': 'Unknown error occurred'}), 500    

if __name__ == '__main__':
     app.run(debug=False)
