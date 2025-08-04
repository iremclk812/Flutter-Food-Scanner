import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

# --- Constants ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2 # 20% of the data will be used for validation
TEST_SPLIT = 0.1 # 10% of the original data will be used for testing

# --- 1. Load Data ---
def load_data(csv_path='merge.csv'):
    """Loads image paths and labels from the merged CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"'{csv_path}' not found. Please run 'merge_data.py' first.")
    
    df = pd.read_csv(csv_path)
    
    # Create a numeric label for each unique food class
    df['label_cat'] = df['label'].astype('category').cat.codes
    return df

# --- 2. Create Datasets ---
def create_datasets(df):
    """Splits data and creates TensorFlow datasets for training, validation, and testing."""
    # Create a mapping from numeric label back to food name
    label_map = dict(enumerate(df['label'].astype('category').cat.categories))
    num_classes = len(label_map)
    print(f"Found {num_classes} food classes.")

    # Split data into training+validation and test sets
    train_val_df, test_df = train_test_split(
        df, 
        test_size=TEST_SPLIT, 
        random_state=42, 
        stratify=df['label_cat'] # Ensure proportional representation of labels
    )

    # Split training+validation into separate training and validation sets
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT), # Adjust split percentage
        random_state=42, 
        stratify=train_val_df['label_cat']
    )

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # Create tf.data.Dataset objects
    train_ds = tf.data.Dataset.from_tensor_slices((train_df['image_path'].values, train_df['label_cat'].values))
    val_ds = tf.data.Dataset.from_tensor_slices((val_df['image_path'].values, val_df['label_cat'].values))
    test_ds = tf.data.Dataset.from_tensor_slices((test_df['image_path'].values, test_df['label_cat'].values))

    return train_ds, val_ds, test_ds, num_classes, label_map

# --- 3. Preprocessing ---
def preprocess_image(image_path, label):
    """Loads, decodes, and resizes an image. One-hot encodes the label."""
    # Load image
    img = tf.io.read_file('images/' + image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    # Normalize pixel values to [0, 1]
    img = img / 255.0
    return img, label

def configure_dataset(ds, num_classes):
    """Applies preprocessing, shuffling, and batching to a dataset."""
    ds = ds.map(lambda x, y: (preprocess_image(x, y)), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

# --- 4. Build Model ---
def build_model(num_classes):
    """Builds a classification model using MobileNetV2 for transfer learning."""
    # Load pre-trained MobileNetV2 model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False, # Don't include the final classification layer
        weights='imagenet'
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create the new model on top
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x) # Regularization
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model

# --- Main Execution ---
if __name__ == '__main__':
    # Load the data from the CSV
    dataframe = load_data()
    
    # Create the datasets
    train_dataset, val_dataset, test_dataset, num_classes, label_map = create_datasets(dataframe)
    
    # Configure the datasets for performance
    train_ds = configure_dataset(train_dataset, num_classes)
    val_ds = configure_dataset(val_dataset, num_classes)
    test_ds = configure_dataset(test_dataset, num_classes)

    print("\nDatasets are ready.")

    # Build the model
    print("Building model...")
    model = build_model(num_classes)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy', # Use this for integer labels
        metrics=['accuracy']
    )
    
    print("Model built and compiled.")
    model.summary()

    # Train the model
    print("\nStarting model training...")
    # Define a callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        filepath='food_classifier.keras',
        save_best_only=False, # Save model at the end of every epoch
        monitor='val_accuracy', # Monitor validation accuracy
        mode='max',
        verbose=1
    )

    history = model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
        callbacks=[checkpoint_callback] # Add the callback here
    )
    
    print("\nTraining finished.")

    # The model is already saved by the callback, so we can skip this.
    # print("Model saved to 'food_classifier.keras'")

    # Evaluate on the test set
    print("\nEvaluating on the test set...")
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Accuracy: {accuracy:.2f}")
