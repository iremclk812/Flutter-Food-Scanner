import os
import io
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import google.generativeai as genai
import json
import re

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Constants ---
IMG_SIZE = (224, 224)
MODEL_PATH = 'food_classifier.keras'
CSV_PATH = os.path.join('archive', 'nutrition.csv')

# --- Global Variables ---
model = None
label_map = None
nutrition_df = None

# --- Gemini API Configuration ---
API_KEY = 'AIzaSyAxR1QTQT7KH31-JNKJ_Iz1LPTN5f3JNfE' 
genai.configure(api_key=API_KEY)

# --- Helper Function ---
def normalize_text(text):
    """Converts text to a standard format for matching (lowercase, no Turkish chars, spaces to underscores)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    replacements = {
        'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c',
        ' ': '_'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

# --- Data Loading and Preprocessing ---
def load_model_and_data():
    """Loads the Keras model and nutrition data, creating the label map and a normalized nutrition lookup table."""
    global model, label_map, nutrition_df

    # 1. Load Model
    print("Loading Keras model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")

    # 2. Load Data
    print(f"Loading nutrition data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    df.rename(columns={'Name': 'label'}, inplace=True)

    # 3. Create Label Map from the definitive labels file
    print("Loading labels from labels.txt...")
    with open('labels.txt', 'r') as f:
        unique_labels = [line.strip() for line in f.readlines()]
    label_map = {i: label for i, label in enumerate(unique_labels)}
    print(f"Label map created with {len(label_map)} classes from labels.txt.")

    # 4. Create Nutrition Lookup Table with a NORMALIZED index
    nutrition_df = df.drop_duplicates(subset=['label']).copy()
    normalized_index = nutrition_df['label'].apply(normalize_text)
    nutrition_df.set_index(normalized_index, inplace=True)
    print("Nutrition lookup table created successfully with normalized index.")
    print(nutrition_df.head())

def preprocess_image(image_bytes):
    """Preprocesses the input image for prediction."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array

# --- Flask Routes ---
@app.route('/predict', methods=['POST'])
def predict():
    """Receives an image, predicts the food, consults Gemini if needed, and returns nutrition info."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        predictions = model.predict(processed_image)
        
        confidence = float(np.max(predictions[0]))
        predicted_class_name = label_map[np.argmax(predictions[0])]
        
        final_prediction_name = predicted_class_name
        gemini_suggestion_text = None

        if confidence < 0.40:
            try:
                print("Confidence below 40%, querying Gemini...")
                gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                img_for_gemini = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                response = gemini_model.generate_content(["What food is in this image? Provide only the name of the food.", img_for_gemini])
                gemini_suggestion_text = response.text.strip()
                print(f"Gemini suggestion: {gemini_suggestion_text}")
                # The final prediction becomes Gemini's suggestion
                final_prediction_name = gemini_suggestion_text
            except Exception as gemini_e:
                print(f"An error occurred with Gemini API: {gemini_e}")

        # Use the normalized version of the final name for lookup
        lookup_key = normalize_text(final_prediction_name)
        
        nutrition_info = {}
        if lookup_key in nutrition_df.index:
            nutrition_info = nutrition_df.loc[lookup_key].to_dict()
            print(f"Nutrition info found locally for '{lookup_key}'.")
        else:
            print(f"Nutrition info NOT found locally for '{lookup_key}'. Querying Gemini...")
            try:
                # Ask Gemini for nutrition data in a structured format
                gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = (
                    f"Provide typical nutritional information for 100g of {final_prediction_name}. "
                    f"Return the response as a JSON object with keys 'calories', 'protein', 'carbohydrates', and 'fats'. "
                    f"Example: {{'calories': 150, 'protein': 10, 'carbohydrates': 5, 'fats': 8}}"
                )
                response = gemini_model.generate_content(prompt)
                
                # First, clean the response text from common markdown artifacts
                cleaned_text = response.text.strip()
                cleaned_text = cleaned_text.replace('```json', '').replace('```', '')

                # Then, use regex to find the JSON object within the cleaned text
                match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    nutrition_info = json.loads(json_str)
                    print(f"Successfully parsed nutrition info from Gemini for '{final_prediction_name}'.")
                else:
                    print(f"Could not find a valid JSON object in Gemini's response: {response.text}")
                    nutrition_info = {}

            except Exception as gemini_nutrition_e:
                print(f"Failed to get or parse nutrition info from Gemini: {gemini_nutrition_e}")
                nutrition_info = {} # Reset to empty if Gemini fails

        # --- New Feature: Get Food Components from Gemini ---
        components = []
        try:
            print("Querying Gemini for food components...")
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            img_for_gemini = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            component_prompt = (
                "List all the distinct food items you see in this image. "
                "Return the response as a JSON array of strings. "
                "For example: [\"macaroni and cheese\", \"green beans\", \"meatloaf\"]"
            )
            
            component_response = gemini_model.generate_content([component_prompt, img_for_gemini])
            
            # Robust parsing for the components list
            cleaned_text = component_response.text.strip().replace('```json', '').replace('```', '')
            match = re.search(r'\[.*\]', cleaned_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                components = json.loads(json_str)
                print(f"Successfully parsed components from Gemini: {components}")
            else:
                print(f"Could not find a valid JSON array in Gemini's component response: {component_response.text}")

        except Exception as gemini_components_e:
            print(f"Failed to get components from Gemini: {gemini_components_e}")


        response_data = {
            'prediction': final_prediction_name, # Always return the human-readable name
            'confidence': f'{confidence:.2%}',
            'nutrition': nutrition_info,
            'components': components, # Add the new key for the frontend
        }

        if gemini_suggestion_text:
            response_data['gemini_suggestion'] = gemini_suggestion_text

        return jsonify(response_data)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# --- Main Execution ---
if __name__ == '__main__':
    load_model_and_data()
    app.run(host='0.0.0.0', port=5000, debug=True)