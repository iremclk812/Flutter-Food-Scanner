import os
import sys
import pandas as pd
import tensorflow as tf
import numpy as np

# --- Constants ---
IMG_SIZE = (224, 224)
MODEL_PATH = 'food_classifier.keras'

def get_label_map(csv_path='merge.csv'):
    """Creates a mapping from numeric label back to food name."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"'{csv_path}' not found. Please run 'merge_data.py' first.")
    
    df = pd.read_csv(csv_path)
    df['label_cat'] = df['label'].astype('category').cat.codes
    label_map = dict(enumerate(df['label'].astype('category').cat.categories))
    return label_map

def preprocess_single_image(image_path):
    """Loads, decodes, and resizes a single image for prediction."""
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0  # Normalize to [0, 1]
        # Add a batch dimension
        img = tf.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# --- Main Execution ---
if __name__ == '__main__':
    # Check for image path argument
    if len(sys.argv) < 2:
        print("Usage: python predict_single_image.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Validate paths
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        sys.exit(1)

    print("--- Single Image Prediction ---")
    try:
        # 1. Load Label Map
        print("Loading class labels...")
        label_map = get_label_map()

        # 2. Load Model
        print(f"Loading model from '{MODEL_PATH}'...")
        model = tf.keras.models.load_model(MODEL_PATH)

        # 3. Preprocess Image
        print(f"Processing image: {os.path.basename(image_path)}...")
        processed_image = preprocess_single_image(image_path)

        if processed_image is not None:
            # 4. Make Prediction
            print("Making prediction...")
            predictions = model.predict(processed_image)
            
            # 5. Decode Prediction
            predicted_class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            predicted_class_name = label_map[predicted_class_index]

            print("--- Prediction Result ---")
            print(f"Predicted Food: {predicted_class_name}")
            print(f"Confidence    : {confidence:.2%}")

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
