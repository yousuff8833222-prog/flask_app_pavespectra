import tensorflow as tf
import os

def convert_to_tflite():
    # Determine the directory where this script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    h5_model_path = os.path.join(SCRIPT_DIR, 'model.h5')
    tflite_model_path = os.path.join(SCRIPT_DIR, 'model.tflite')

    if not os.path.exists(h5_model_path):
        print(f"Error: Model file '{h5_model_path}' not found. Please train the model first.")
        return

    print(f"Loading Keras model from: {h5_model_path}")
    model = tf.keras.models.load_model(h5_model_path)

    # Initialize the TFLite converter
    print("Initializing TFLite Converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optimization: Dynamic range quantization
    # This reduces weight precision to 8-bits, making the model much smaller
    # and faster on edge device CPUs.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    print("Converting model (this may take a moment)...")
    tflite_model = converter.convert()

    print(f"Saving TFLite model to: {tflite_model_path}")
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print("Conversion successful!")

if __name__ == "__main__":
    convert_to_tflite()