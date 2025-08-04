import os
import io
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Constants ---
IMG_SIZE = (224, 224)
MODEL_PATH = 'food_classifier.keras'
CSV_PATH = 'merge.csv'  # This is the single source of all data

# --- Global Variables ---
model = None
label_map = None
nutrition_df = None

def load_model_and_data():
    """Loads the Keras model, labels, and all nutrition data from merge.csv."""
    global model, label_map, nutrition_df
    
    print("Loading Keras model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")

    print(f"Loading all data from {CSV_PATH}...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    
    # 1. Create label mapping for the model's output
    # This uses the 'label' column to create a map from number to category name
    df['label_cat'] = df['label'].astype('category').cat.codes
    label_map = dict(enumerate(df['label'].astype('category').cat.categories))
    print("Labels loaded successfully.")

    # 2. Create the nutrition lookup table from the same CSV
    # Dynamically determine nutrition columns by excluding known non-nutrition columns
    all_columns = df.columns.tolist()
    non_nutrition_cols = ['label', 'image_path', 'weight', 'label_cat']
    nutrition_cols = [col for col in all_columns if col not in non_nutrition_cols]
    
    # The columns for the final dataframe will be the label + all nutrition columns
    lookup_cols = ['label'] + nutrition_cols
            
    nutrition_df = df[lookup_cols].drop_duplicates(subset=['label']).set_index('label')
    print("Nutrition lookup table created successfully with all available nutrition columns.")
    print("Columns included:", nutrition_df.columns.tolist())
    print("Nutrition data preview:")
    print(nutrition_df.head())


def preprocess_image(image_bytes):
    """Preprocesses the input image for prediction."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    """Receives an image, predicts the food class, and returns the result with all nutrition info."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        predictions = model.predict(processed_image)
        
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        predicted_class_name = label_map[predicted_class_index]
        
        # Get nutrition info using the predicted class name as the key
        if predicted_class_name in nutrition_df.index:
            nutrition_info = nutrition_df.loc[predicted_class_name].to_dict()
        else:
            nutrition_info = {}

        return jsonify({
            'prediction': predicted_class_name,
            'confidence': f'{confidence:.2%}',
            'nutrition': nutrition_info
        })

    except Exception as e:
        print(f"An error occurred: {e}") # Log the error for debugging
        return jsonify({'error': str(e)}), 500

# --- Main Execution ---
if __name__ == '__main__':
    load_model_and_data()
    app.run(host='0.0.0.0', port=5000, debug=True)