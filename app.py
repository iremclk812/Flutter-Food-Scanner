import os
import io
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import google.generativeai as genai

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Constants ---
IMG_SIZE = (224, 224)
MODEL_PATH = 'food_classifier.keras'
CSV_PATH = os.path.join('archive', 'nutrition.csv') # Use the actual nutrition data source

# --- Global Variables ---
model = None
label_map = None
nutrition_df = None

# --- Gemini API Configuration ---
# IMPORTANT: Replace "YOUR_API_KEY" with your actual Google AI Studio API key.
# For better security, it's recommended to load the key from an environment variable.
API_KEY = 'AIzaSyAxR1QTQT7KH31-JNKJ_Iz1LPTN5f3JNfE' 
genai.configure(api_key=API_KEY)


def load_model_and_data():
    """Loads the Keras model and nutrition data, creating the label map directly."""
    global model, label_map, nutrition_df

    # 1. Load the Keras model
    print("Loading Keras model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")

    # 2. Load nutrition data from the source CSV
    print(f"Loading nutrition data from {CSV_PATH}...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    # Standardize column name to 'label' to match training logic
    df.rename(columns={'Name': 'label'}, inplace=True)

    # 3. Create the label map based on alphabetical order of unique labels
    # This is critical because the model was trained with labels sorted alphabetically
    # (as implicitly done by `astype('category').cat.categories` in train_cnn.py)
    unique_labels = sorted(df['label'].unique())
    label_map = {i: label for i, label in enumerate(unique_labels)}
    print(f"Label map created successfully with {len(label_map)} classes.")

    # 4. Create the nutrition lookup table
    # Use the first entry for each food type as the definitive nutrition source
    nutrition_df = df.drop_duplicates(subset=['label']).set_index('label')
    print("Nutrition lookup table created successfully.")
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
        
        gemini_suggestion = None
        # If confidence is below 40%, ask Gemini for a second opinion
        if confidence < 0.40:
            try:
                print("Confidence below 40%, querying Gemini...")
                # Use the gemini-1.5-flash model for speed and cost-effectiveness
                gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                
                # Prepare the image for Gemini
                img_for_gemini = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                # Send the image and prompt to Gemini
                response = gemini_model.generate_content(["What food is in this image? Provide only the name of the food.", img_for_gemini])
                
                # Clean up the response to get just the food name
                gemini_suggestion = response.text.strip()
                print(f"Gemini suggestion: {gemini_suggestion}")

            except Exception as gemini_e:
                print(f"An error occurred with Gemini API: {gemini_e}")
                # If Gemini fails, we still proceed without its suggestion
                gemini_suggestion = "Error: Could not get suggestion."

        # Get nutrition info using the original predicted class name as the key
        if predicted_class_name in nutrition_df.index:
            nutrition_info = nutrition_df.loc[predicted_class_name].to_dict()
        else:
            nutrition_info = {}

        # Prepare the final JSON response
        response_data = {
            'prediction': predicted_class_name,
            'confidence': f'{confidence:.2%}',
            'nutrition': nutrition_info
        }

        # Add Gemini's suggestion to the response if it exists
        if gemini_suggestion:
            response_data['gemini_suggestion'] = gemini_suggestion

        return jsonify(response_data)

    except Exception as e:
        print(f"An error occurred: {e}") # Log the error for debugging
        return jsonify({'error': str(e)}), 500

# --- Main Execution ---
if __name__ == '__main__':
    load_model_and_data()
    app.run(host='0.0.0.0', port=5000, debug=True)