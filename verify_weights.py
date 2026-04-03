import os
import numpy as np
from tensorflow.keras.models import load_model

# Define the absolute path to the model file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(SCRIPT_DIR, 'sample.keras')

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
else:
    print(f"Loading model from: {model_path}")
    try:
        # Load the full model
        model = load_model(model_path)
        
        print("\n--- Model Summary ---")
        model.summary()
        
        print("\n--- Layer Weight Verification ---")
        for i, layer in enumerate(model.layers):
            weights = layer.get_weights()
            if len(weights) > 0:
                # weights[0] is typically the kernel/filter weights
                # weights[1] is typically the bias vector
                w = weights[0]
                print(f"Layer {i} ({layer.__class__.__name__} - {layer.name}):")
                print(f"  - Weights Shape:   {w.shape}")
                print(f"  - Weights Mean:    {np.mean(w):.8f}")
                print(f"  - Weights Std Dev: {np.std(w):.8f}")
                print(f"  - Non-zero Count:  {np.count_nonzero(w)} / {w.size}")
            else:
                print(f"Layer {i} ({layer.__class__.__name__} - {layer.name}): No trainable weights.")
    except Exception as e:
        print(f"Failed to verify model: {e}")