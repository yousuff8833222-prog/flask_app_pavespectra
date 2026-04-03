import os
# Suppress TensorFlow GPU and optimization warnings before importing TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import cv2
import logging
import time
import threading
import imutils
import random
from collections import deque
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Required for non-interactive plotting in threads
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, Response, jsonify, url_for
import tensorflow as tf

# --- GPU Configuration ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"Flask App: GPU enabled with memory growth ({len(gpus)} device(s))")
    except RuntimeError as e:
        logging.error(f"Flask App: GPU Config Error: {e}")
# -------------------------
from whitenoise import WhiteNoise
from tensorflow.keras.models import load_model
from sklearn.ensemble import IsolationForest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Try to import winsound for Windows alerts
try:
    import winsound
except ImportError:
    winsound = None

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class CameraStream:
    """Threaded camera stream for consistent FPS and thread-safety."""
    def __init__(self, src=0):
        self.lock = threading.Lock()
        self.stopped = False
        self.grabbed = False
        self.frame = None

        # Try primary source (CAP_DSHOW is often better on Windows)
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.stream.isOpened():
            self.stream = cv2.VideoCapture(src)
            
        (self.grabbed, self.frame) = self.stream.read()

        # If first source failed to open or grab a frame, try secondary
        if not self.grabbed or not self.stream.isOpened():
            if self.stream.isOpened():
                self.stream.release()
            
            logging.info("Source 0 failed, trying Source 1...")
            self.stream = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            if not self.stream.isOpened():
                self.stream = cv2.VideoCapture(1)
            (self.grabbed, self.frame) = self.stream.read()

        if not self.grabbed:
            logging.error("CameraStream: Could not grab initial frame from any source.")

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if self.stream.isOpened():
                (grabbed, frame) = self.stream.read()
                with self.lock:
                    self.grabbed, self.frame = grabbed, frame
                time.sleep(0.01) # Brief sleep to reduce CPU usage
            else:
                break

    def read(self):
        with self.lock:
            return self.frame.copy() if (self.grabbed and self.frame is not None) else None

    def stop(self):
        self.stopped = True
        self.stream.release()

# Initialize Flask.
app = Flask(__name__, template_folder='templates', static_folder='static')
app.wsgi_app = WhiteNoise(app.wsgi_app, root=os.path.join(SCRIPT_DIR, 'static/'), prefix='static/')

# Support multiple potential model filenames
POSSIBLE_MODEL_PATHS = [
    os.path.join(SCRIPT_DIR, 'model.h5'),   # Absolute path (calculated dynamically)
    os.path.join(SCRIPT_DIR, 'sample.keras'),
    'model.h5',                            # Simple relative path (as requested)
    'sample.keras'                         # Simple relative path (as requested)
]

# Global State & Locks
lock = threading.Lock()
data_lock = threading.Lock()
is_camera_on = True
detection_enabled = True
model = None
input_size = 224
has_model = False

def load_system_model():
    global model, input_size, has_model
    
    selected_path = None
    for path in POSSIBLE_MODEL_PATHS:
        if os.path.exists(path):
            selected_path = path
            break
            
    if not selected_path:
        logging.error("No model file found (checked model.h5, sample.keras). Detection disabled.")
        return False

    try:
        model = load_model(selected_path, compile=False)
        input_size = model.input_shape[1]
        has_model = True
        logging.info(f"Model loaded successfully. Input size: {input_size}")
        return True
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return False

# Initialization
load_system_model()
camera_stream = None
if is_camera_on:
    camera_stream = CameraStream().start()

# Analytics State
severity_data = deque(maxlen=1000)  # Keep only the last 1000 records to prevent memory issues
heatmap_coords = deque(maxlen=1000) # Memory-safe heatmap coordinates

class PotholeDetector:
    """Handles model inference and data logging logic."""
    @staticmethod
    def get_mock_gps():
        """Simulates GPS coordinates for log entries."""
        lat = 12.9716 + random.uniform(-0.0005, 0.0005)
        lon = 77.5946 + random.uniform(-0.0005, 0.0005)
        return round(lat, 6), round(lon, 6)

    @staticmethod
    def predict_batch(rois, model_obj, size):
        if not rois or model_obj is None: return []
        try:
            batch = np.array([cv2.resize(r, (size, size)) for r in rois]).astype('float32') / 255.0
            predictions = model_obj.predict(batch, verbose=0)
            return [(np.argmax(p), np.max(p)) for p in predictions]
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return []

    @staticmethod
    def log_detection(confidence, x_pos, y_pos):
        lat, lon = PotholeDetector.get_mock_gps()
        with data_lock:
            severity_data.append({
                'timestamp': time.strftime("%H:%M:%S"),
                'confidence': float(confidence),
                'lat': lat,
                'lon': lon,
                'x_pos': x_pos
            })
            heatmap_coords.append((x_pos, y_pos))

def play_alert_sound():
    """Triggers a beep on Windows systems."""
    if winsound:
        try:
            winsound.Beep(800, 200) # Softer pitch and shorter duration to reduce annoyance
        except Exception:
            pass

def generate_frames():
    global camera_stream
    last_alert_time = 0
    last_results = []
    roi_x_positions = [0, 0, 0] # Initialize with defaults
    frame_count = 0
    height, width = 480, 700 # Default dimensions for placeholders
    roi_h = 240
    top = 160
    bottom = 400

    while True:
        display_frame = None
        frame = None

        if not is_camera_on:
            # Generate a "Paused" frame so the stream doesn't hang
            display_frame = np.zeros((480, 700, 3), dtype=np.uint8)
            cv2.putText(display_frame, "System Paused", (230, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            time.sleep(0.2)
        else:
            with lock:
                if camera_stream is not None:
                    frame = camera_stream.read()

            if frame is None:
                # Generate an error frame if camera is on but failing
                display_frame = np.zeros((480, 700, 3), dtype=np.uint8)
                cv2.putText(display_frame, "Camera Error: No Feed Detected", (140, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                time.sleep(0.5) 
            else:
                # Process valid frame
                try:
                    frame = imutils.resize(frame, width=700)
                    frame = cv2.flip(frame, 1)
                    display_frame = frame.copy()
                    (height, width) = frame.shape[:2]
                except Exception as e:
                    logging.error(f"Frame processing error: {e}")
                    display_frame = np.zeros((480, 700, 3), dtype=np.uint8)
                    cv2.putText(display_frame, "Processing Error", (240, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Perform detection only if we have a valid camera frame
        if frame is not None and detection_enabled and has_model:
            # Calculate dynamic scanning parameters to match realtimePredictor.py
            roi_h = int(height * 0.5) 
            top = int(height * 0.35)
            bottom = top + roi_h
            roi_x_positions = [int(width * 0.05), int(width * 0.35), int(width * 0.65)]

            # Initialize results list if empty
            if not last_results or len(last_results) != len(roi_x_positions):
                last_results = [(0, 0.0)] * len(roi_x_positions)

            if frame_count % 2 == 0:
                try:
                    rois = [frame[top:bottom, x:x+roi_h] for x in roi_x_positions]
                    # Filter out empty ROIs
                    valid_rois = [r for r in rois if r.size > 0]
                    if valid_rois:
                        last_results = PotholeDetector.predict_batch(valid_rois, model, input_size)
                except Exception as e:
                    logging.error(f"Prediction batch error: {e}")
            
            # Display results from the last successful prediction
            for i in range(min(len(last_results), len(roi_x_positions))):
                p_class, p_prob = last_results[i]
                x_start = roi_x_positions[i]
                try:
                    if p_class == 1 and p_prob > 0.75:
                        now = time.time()
                        if now - last_alert_time > 8.0: # Increased cooldown to avoid annoyance
                            threading.Thread(target=play_alert_sound, daemon=True).start()
                            last_alert_time = now
                        
                        PotholeDetector.log_detection(p_prob, x_start + roi_h//2, top + roi_h//2)

                        cv2.rectangle(display_frame, (x_start, top), (x_start+roi_h, bottom), (0, 0, 255), 3)
                        cv2.putText(display_frame, f"POTHOLE: {p_prob*100:.1f}%", (x_start, top - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(display_frame, (x_start, top), (x_start+roi_h, bottom), (0, 255, 0), 1)
                except Exception as e:
                    logging.debug(f"Error processing ROI {i}: {e}")
            
            frame_count += 1

        # Final safety check before encoding
        if display_frame is not None:
            ret, buffer = cv2.imencode('.jpg', display_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                logging.error("Failed to encode frame.")
        else:
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_camera')
def toggle_camera():
    global is_camera_on, camera_stream
    with lock:
        is_camera_on = not is_camera_on
        if not is_camera_on and camera_stream:
            camera_stream.stop()
            camera_stream = None
        elif is_camera_on and camera_stream is None:
            camera_stream = CameraStream().start()
    return jsonify({"status": is_camera_on})

@app.route('/toggle_detection')
def toggle_detection():
    global detection_enabled
    detection_enabled = not detection_enabled
    return jsonify({"status": detection_enabled})

@app.route('/log_data')
def log_data():
    with data_lock:
        if not severity_data:
            return jsonify([])
        
        df = pd.DataFrame(severity_data).tail(20) # Latest 20 records
    
    if len(df) > 5:
        try:
            iso = IsolationForest(contamination=0.1, random_state=42)
            df['anomaly'] = iso.fit_predict(df[['confidence']])
            df['status'] = df['anomaly'].apply(lambda x: 'ANOMALY' if x == -1 else 'STABLE')
        except Exception as e:
            logging.error(f"Anomaly detection failed: {e}")
            df['status'] = 'ERROR'
    else:
        df['status'] = 'COLLECTING'
        
    return jsonify(df.to_dict(orient='records'))

@app.route('/heatmap')
def heatmap():
    with data_lock:
        coords = list(heatmap_coords)

    fig, ax = plt.subplots(figsize=(10, 3), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    
    if coords:
        x, y = zip(*coords)
        hb = ax.hexbin(x, y, gridsize=15, cmap='magma', mincnt=1)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Frequency', color='white')
        cb.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
    
    ax.set_title("Spatial Detection Density (X-Y Plane)", color='white', pad=10)
    ax.axis('off')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', facecolor='#0d1117', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{plot_url}"

@app.context_processor
def override_url_for():
    """Automates cache busting by appending a timestamp to static file URLs."""
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.static_folder, filename)
            if os.path.exists(file_path):
                values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

if __name__ == '__main__':
    # Use Railway's port if available, otherwise default to 5000 for your laptop
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)