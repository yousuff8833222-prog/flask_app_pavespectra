import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization, RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
import cv2, glob, os

global size

# --- GPU Configuration ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) detected and configured: {len(gpus)}")
    except RuntimeError as e:
        print(f"GPU Configuration Error: {e}")
# -------------------------

# Determine the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_transfer_model(input_shape):
    # Transfer Learning with MobileNetV2 for high accuracy
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = Sequential([
        tf.keras.layers.Input(shape=input_shape),
        RandomFlip("horizontal"),
        RandomRotation(0.15),
        RandomZoom(0.15),
        RandomBrightness(0.2),
        RandomContrast(0.2),
        base_model,
        GlobalAveragePooling2D(),
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),
        Dense(2, activation='softmax')
    ])
    return model, base_model

size = 224 # Standard input size for MobileNetV2

def load_images_from_folder(folder_path, size, extensions=['jpg', 'jpeg', 'png']):
    images = []
    for ext in extensions:
        files = glob.glob(os.path.join(folder_path, f"*.{ext}"))
        for img_path in files:
            # Load in BGR (3 channels) for Transfer Learning
            img_data = cv2.imread(img_path)
            if img_data is not None:
                img_data = cv2.resize(img_data, (size, size))
                images.append(img_data)
    return np.asarray(images, dtype=np.float32)

print("Loading datasets...")
## load Training data
temp1 = load_images_from_folder(os.path.join(SCRIPT_DIR, "My Dataset", "train", "Pothole"), size)
temp2 = load_images_from_folder(os.path.join(SCRIPT_DIR, "My Dataset", "train", "Plain"), size)

## load Testing data
temp4 = load_images_from_folder(os.path.join(SCRIPT_DIR, "My Dataset", "test", "Plain"), size)
temp3 = load_images_from_folder(os.path.join(SCRIPT_DIR, "My Dataset", "test", "Pothole"), size)

print(f"Training images: Potholes={len(temp1)}, Plain={len(temp2)}")
print(f"Testing images: Potholes={len(temp3)}, Plain={len(temp4)}")

if len(temp1) == 0 or len(temp2) == 0:
    raise ValueError("Error: Training dataset is empty. Check your folder paths and ensure images exist in 'My Dataset/train/'.")


X_train = np.concatenate([temp1, temp2], axis=0)
X_test = np.concatenate([temp3, temp4], axis=0)

# Normalize pixel values to [0, 1] range
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train1 = np.ones([len(temp1)], dtype=int)
y_train2 = np.zeros([len(temp2)], dtype=int)
y_test1 = np.ones([len(temp3)], dtype=int)
y_test2 = np.zeros([len(temp4)], dtype=int)

y_train = np.concatenate([y_train1, y_train2], axis=0)
y_test = np.concatenate([y_test1, y_test2], axis=0)

X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_test, y_test = shuffle(X_test, y_test, random_state=42)

# Reshape for 3 channels (BGR)
X_train = X_train.reshape(X_train.shape[0], size, size, 3)
X_test = X_test.reshape(X_test.shape[0], size, size, 3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


print("train shape X", X_train.shape)
print("train shape y", y_train.shape)

inputShape = (size, size, 3)
model, base_model = create_transfer_model(inputShape)

# Phase 1: Train Top Layers
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

print("Starting Phase 1 Training...")
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stop, reduce_lr])

# Phase 2: Fine-tuning (Unfreeze part of base model)
print("Starting Phase 2: Fine-tuning...")
base_model.trainable = True
# Only unfreeze top layers of MobileNetV2
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stop, reduce_lr])

metrics = model.evaluate(X_test, y_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))

print("Saving model weights and configuration file")

model.save(os.path.join(SCRIPT_DIR, 'model.h5'))
model.save(os.path.join(SCRIPT_DIR, 'model.keras'))

print(f"Saved model to {SCRIPT_DIR} in both .h5 and .keras formats.")

# --- Accuracy & Loss Visualization ---
def plot_training_curves(history):
    """Plots the accuracy and loss curves to check for overfitting."""
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'training_performance.png'))
    plt.show()

plot_training_curves(history)