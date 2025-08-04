import gradio as gr
import tensorflow as tf
import pandas as pd
import numpy as np

# --- Constants ---
MODEL_PATH = 'food_classifier.keras'
NUTRITION_CSV_PATH = 'merge.csv'
IMG_SIZE = (224, 224)

# --- Load Model and Data ---
try:
    print("Loading model and data...")
    model = tf.keras.models.load_model(MODEL_PATH)
    nutrition_df = pd.read_csv(NUTRITION_CSV_PATH)
    
    # Create the inverse label map to convert model output to food names
    labels = sorted(nutrition_df['label'].unique())
    inverse_label_map = {i: label for i, label in enumerate(labels)}
    print("Model and data loaded successfully.")
except Exception as e:
    print(f"Error loading model or data: {e}")
    model = None # Set model to None if loading fails

def predict_food(image):
    """Takes a user-uploaded image, preprocesses it, and returns the prediction."""
    if model is None:
        return "Model is not loaded. Please check the console for errors.", None

    # Preprocess the image
    img = tf.image.resize(image, IMG_SIZE)
    img = img / 255.0
    img_array = tf.expand_dims(img, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_label = inverse_label_map[predicted_index]
    confidence = np.max(predictions) * 100

    # Get nutrition info
    nutrition_info = nutrition_df[nutrition_df['label'] == predicted_label].iloc[0]
    
    # Format output
    food_name = predicted_label.replace('_', ' ').title()
    confidence_text = f"{confidence:.2f}% confidence"
    
    nutrition_output = (
        f"Calories: {nutrition_info['Calories']:.2f} kcal\n"
        f"Protein: {nutrition_info['Protein(g)']:.2f} g\n"
        f"Fat: {nutrition_info['Fat(g)']:.2f} g\n"
        f"Carbs: {nutrition_info['Carbs(g)']:.2f} g"
    )
    
    return f"{food_name} ({confidence_text})", nutrition_output

# --- Create Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🍔 Food Image Calorie Recognition 🥗")
    gr.Markdown("Upload an image of a food item, and the model will predict what it is and its nutritional information per 100g.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Food Image")
            submit_btn = gr.Button("Predict")
        with gr.Column():
            label_output = gr.Label(label="Predicted Food")
            nutrition_output = gr.Textbox(label="Nutritional Information (per 100g)", lines=4)
            
    submit_btn.click(
        fn=predict_food, 
        inputs=image_input, 
        outputs=[label_output, nutrition_output]
    )

# --- Launch the App ---
if __name__ == "__main__":
    if model is not None:
        demo.launch()
    else:
        print("Application cannot start because the model failed to load.")

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob

# Load nutrition data
nutrition_df = pd.read_csv(os.path.join('archive', 'nutrition.csv'))
# Calculate average calories for each label
calorie_lookup = nutrition_df.groupby('label')['calories'].mean().to_dict()

sample_imgs = 25

# Correct the input path to look inside the 'archive' directory for h5 files
for c_path in glob(os.path.join('archive', 'food_*.h5')):
    try:
        with h5py.File(c_path, 'r') as n_file:
            print(f"Processing file: {c_path}")
            if 'images' not in n_file or 'category' not in n_file or 'category_names' not in n_file:
                print(f"Skipping {c_path}: missing required datasets.")
                continue

            total_imgs = n_file['images'].shape[0]
            if total_imgs == 0:
                print(f"Skipping {c_path}: no images found.")
                continue
                
            read_idxs = slice(0, min(sample_imgs, total_imgs))
            im_data = n_file['images'][read_idxs]
            im_label = n_file['category'][read_idxs]
            label_names = [x.decode() for x in n_file['category_names'][:] ]

            fig, m_ax = plt.subplots(5, 5, figsize=(12, 12))
            fig.suptitle(os.path.basename(c_path), fontsize=16)
            
            for c_ax, c_label, c_img in zip(m_ax.flatten(), im_label, im_data):
                label_name = label_names[np.argmax(c_label)]
                calories = calorie_lookup.get(label_name, 'N/A')
                title = f'{label_name}\nCalories: {calories:.0f}' if isinstance(calories, (int, float)) else f'{label_name}\nCalories: N/A'
                
                c_ax.imshow(c_img if c_img.shape[2] == 3 else c_img[:, :, 0], cmap='gray')
                c_ax.axis('off')
                c_ax.set_title(title)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            output_filename = f'overview_{os.path.basename(c_path)}.png'
            fig.savefig(output_filename)
            print(f"Saved overview to {output_filename}")
            plt.close(fig) # Close the figure to free memory

    except Exception as e:
        print(f"Could not process file {c_path}. Error: {e}")
