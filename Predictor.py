import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load without compiling to silence warnings, then compile manually for evaluation
model_path = os.path.join(SCRIPT_DIR, 'model.h5')
if not os.path.exists(model_path):
    model_path = os.path.join(SCRIPT_DIR, 'sample.keras')

model = load_model(model_path, compile=False)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Dynamically set size based on the model's expected input shape
size = model.input_shape[1]

# X_test = np.load('./models/trainData/128x72x3x10000/X_test.npy')
# y_test = np.load('./models/trainData/128x72x3x10000/y_test.npy')
def load_test_images(path, size, extensions=['jpg', 'jpeg', 'png']):
    images = []
    for ext in extensions:
        files = glob.glob(os.path.join(path, f"*.{ext}"))
        for img in files:
            img_data = cv2.imread(img)
            if img_data is not None:
                images.append(cv2.resize(img_data, (size, size)))
    if not images:
        return np.empty((0, size, size, 3), dtype=np.float32)
    return np.asarray(images, dtype=np.float32)

## load Testing data : non-pothole
temp4 = load_test_images(os.path.join(SCRIPT_DIR, "My Dataset", "test", "Plain"), size)

## load Testing data : potholes
temp3 = load_test_images(os.path.join(SCRIPT_DIR, "My Dataset", "test", "Pothole"), size)

if len(temp3) == 0 and len(temp4) == 0:
    raise ValueError("No testing images found. Check 'My Dataset/test/' paths.")

X_test = np.concatenate([temp3, temp4], axis=0)
X_test = X_test.reshape(X_test.shape[0], size, size, 3)

y_test1 = np.ones([temp3.shape[0]],dtype = int)
y_test2 = np.zeros([temp4.shape[0]],dtype = int)

y_test = np.concatenate([y_test1, y_test2], axis=0)
y_test = to_categorical(y_test, num_classes=2)

X_test = X_test / 255.0


predictions = model.predict(X_test)
tests = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_true, tests))

print("\n--- Accuracy Guide (Classification Report) ---")
target_names = ['Plain', 'Pothole']
print(classification_report(y_true, tests, target_names=target_names))

metrics = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {metrics[1]*100:.2f}%")