import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import argparse
from tensorflow.keras.models import load_model

def predict_single_image(image_path):
    # Determine the directory where this script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(SCRIPT_DIR, 'model.h5')

    # Normalize the path to handle Windows backslashes and relative paths correctly
    image_path = os.path.normpath(os.path.abspath(image_path))

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at: {image_path}")
        return

    # Load the trained model
    print(f"Loading model...")
    model = load_model(model_path, compile=False)
    
    # Dynamically detect the required input size (e.g., 100 or 300)
    size = model.input_shape[1]

    # Preprocess the image
    # 1. Load in BGR (3 channels) - matching Transfer Learning update
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not decode image.")
        return

    # 2. Resize and normalize
    img_resized = cv2.resize(img, (size, size))
    img_normalized = img_resized.reshape(1, size, size, 3).astype('float32') / 255.0

    # Perform prediction
    prediction = model.predict(img_normalized, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100

    # Map class to label
    labels = {0: "Plain", 1: "Pothole"}
    result = labels.get(predicted_class, "Unknown")

    print("\n--- Prediction Result ---")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Label: {result}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict pothole presence in a single image.")
    parser.add_argument("--image", required=True, help="Path to the image file.")
    args = parser.parse_args()
    predict_single_image(args.image)