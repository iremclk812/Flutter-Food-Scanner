import tensorflow as tf
import pandas as pd
import numpy as np
import argparse

# --- Constants ---
IMG_SIZE = (224, 224)
MODEL_PATH = 'food_classifier.keras'
NUTRITION_CSV_PATH = 'merge.csv'

def load_and_preprocess_image(image_path):
    """Loads and preprocesses an image for model prediction."""
    # Load the image file
    img = tf.io.read_file(image_path)
    # Decode the image to a tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Resize the image
    img = tf.image.resize(img, IMG_SIZE)
    # Rescale pixel values to [0, 1]
    img = img / 255.0
    # Expand dimensions to create a batch of 1
    img = tf.expand_dims(img, axis=0)
    return img

def get_nutrition_info(dataframe, predicted_label):
    """Retrieves nutrition information for a given label from the dataframe."""
    # Find the first row matching the label
    nutrition_info = dataframe[dataframe['label'] == predicted_label].iloc[0]
    return nutrition_info

# --- Main Execution ---
if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Predict food type and calories from an image.')
    parser.add_argument('image_path', type=str, help='Path to the food image.')
    args = parser.parse_args()

    try:
        # --- 1. Load Model and Data ---
        print("Loading model and data...")
        model = tf.keras.models.load_model(MODEL_PATH)
        nutrition_df = pd.read_csv(NUTRITION_CSV_PATH)

        # Recreate the label mapping from the training data
        # Note: It's crucial that this matches the mapping used during training.
        labels = sorted(nutrition_df['label'].unique())
        inverse_label_map = {i: label for i, label in enumerate(labels)}
        print("Model and data loaded successfully.")

        # --- 2. Preprocess Image ---
        print(f"\nProcessing image: {args.image_path}")
        processed_image = load_and_preprocess_image(args.image_path)

        # --- 3. Make Prediction ---
        print("Making prediction...")
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_label = inverse_label_map[predicted_index]
        confidence = np.max(predictions) * 100

        print(f"\nPredicted Food: {predicted_label.replace('_', ' ').title()} ({confidence:.2f}% confidence)")

        # --- 4. Get Nutrition Info ---
        nutrition_data = get_nutrition_info(nutrition_df, predicted_label)
        
        print("\n--- Nutrition Information (per 100g) ---")
        print(f"Calories: {nutrition_data['Calories']:.2f} kcal")
        print(f"Protein: {nutrition_data['Protein(g)']:.2f} g")
        print(f"Fat: {nutrition_data['Fat(g)']:.2f} g")
        print(f"Carbohydrates: {nutrition_data['Carbs(g)']:.2f} g")

    except FileNotFoundError:
        print(f"Error: The file '{args.image_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
