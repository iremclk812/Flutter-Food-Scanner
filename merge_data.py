import os
import pandas as pd

# Define paths
images_dir = 'images'
labels_path = os.path.join('meta', 'labels.txt')
nutrition_path = os.path.join('archive', 'nutrition.csv')
output_path = 'merge.csv'

# Function to format labels to match the nutrition CSV (e.g., "Apple pie" -> "apple_pie")
def format_label(label):
    return label.strip().lower().replace(' ', '_')

try:
    # 1. Read the official Food-101 labels from labels.txt
    with open(labels_path, 'r') as f:
        official_labels = {format_label(line) for line in f}

    # 2. Create a list of image paths for the official labels
    image_data = []
    for label in official_labels:
        label_path = os.path.join(images_dir, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                if image_file.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(label, image_file).replace('\\', '/')
                    image_data.append({'label': label, 'image_path': image_path})
    
    images_df = pd.DataFrame(image_data)

    # 3. Read and filter nutrition data for 100g servings
    nutrition_df = pd.read_csv(nutrition_path)
    nutrition_df.rename(columns={'Name': 'label'}, inplace=True)
    # Assuming 100g is a standard measure available for most items
    nutrition_100g_df = nutrition_df[nutrition_df['weight'] == 100].copy()

    # If some labels don't have a 100g entry, we can take the first available entry as a fallback
    missing_labels = official_labels - set(nutrition_100g_df['label'])
    if missing_labels:
        fallback_df = nutrition_df[nutrition_df['label'].isin(missing_labels)].groupby('label').first().reset_index()
        nutrition_100g_df = pd.concat([nutrition_100g_df, fallback_df], ignore_index=True)

    # 4. Merge image data with the filtered nutrition data
    merged_df = pd.merge(images_df, nutrition_100g_df, on='label', how='inner')

    # 5. Reorder and save the final CSV
    cols = ['label', 'image_path'] + [col for col in merged_df.columns if col not in ['label', 'image_path']]
    merged_df = merged_df[cols]

    merged_df.to_csv(output_path, index=False)

    print(f"Successfully merged {len(merged_df)} images from {merged_df['label'].nunique()} official labels.")
    print(f"Merged data saved to '{output_path}'")

except FileNotFoundError as e:
    print(f"Error: {e}. Please make sure the file paths are correct.")
except Exception as e:
    print(f"An error occurred: {e}")
