import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# --- Constants from train_cnn.py ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
MODEL_PATH = 'food_classifier.keras'

# --- Data Loading and Preprocessing Functions from train_cnn.py ---

def load_data(csv_path='merge.csv'):
    """Loads image paths and labels from the merged CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"'{csv_path}' not found. Please ensure it's in the correct directory.")
    
    df = pd.read_csv(csv_path)
    df['label_cat'] = df['label'].astype('category').cat.codes
    return df

def create_test_dataset(df):
    """Splits data and creates a TensorFlow dataset for testing."""
    num_classes = len(df['label'].astype('category').cat.categories)

    # Split data into training+validation and test sets to replicate the original split
    train_val_df, test_df = train_test_split(
        df, 
        test_size=TEST_SPLIT, 
        random_state=42, 
        stratify=df['label_cat']
    )

    print(f"Using {len(test_df)} samples for testing.")

    # Create tf.data.Dataset object for the test set
    test_ds = tf.data.Dataset.from_tensor_slices((test_df['image_path'].values, test_df['label_cat'].values))

    return test_ds, num_classes

def preprocess_image(image_path, label):
    """Loads, decodes, and resizes an image."""
    img = tf.io.read_file('C:\Users\iremc\OneDrive\Resimler\pngtree-hamburger-png-image_13094305.png')
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img, label

def configure_dataset(ds):
    """Applies preprocessing and batching to a dataset."""
    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

# --- Main Execution ---
if __name__ == '__main__':
    print("--- Model Evaluation Script ---")
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please make sure the trained model file exists in the same directory.")
    else:
        try:
            # 1. Load Data
            print("Loading and preparing test data...")
            dataframe = load_data()
            test_dataset, num_classes = create_test_dataset(dataframe)
            
            # 2. Configure Dataset
            test_ds = configure_dataset(test_dataset)
            print("Test data is ready.")

            # 3. Load Trained Model
            print(f"Loading model from '{MODEL_PATH}'...")
            model = tf.keras.models.load_model(MODEL_PATH)
            model.summary()

            # 4. Evaluate on the test set
            print("\nEvaluating model performance on the test set...")
            loss, accuracy = model.evaluate(test_ds)
            
            print("--- Evaluation Complete ---")
            print(f"Test Loss    : {loss:.4f}")
            print(f"Test Accuracy: {accuracy:.4f} ({accuracy:.2%})")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure that 'merge.csv' and the 'images' folder are present.")
