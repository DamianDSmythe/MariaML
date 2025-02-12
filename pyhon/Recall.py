import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'python', 'MariaMLV00006.h5')
TEST_DIR = os.path.join(BASE_DIR, 'test_data')

# Load the trained model
model = load_model(MODEL_PATH)

# Parameters
IMG_HEIGHT = 1080
IMG_WIDTH = 1080

# Function to load and preprocess image
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Classify images in test_data folder
for img_name in os.listdir(TEST_DIR):
    img_path = os.path.join(TEST_DIR, img_name)
    img_array = preprocess_image(img_path)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    # Generic label for predicted class
    breed = f"Class {predicted_class}"
    print(f"Image: {img_name} | Predicted: {breed} | Confidence: {confidence:.2f}")
