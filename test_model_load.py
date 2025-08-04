print("--- Starting imports ---")
import tensorflow as tf
print("--- TensorFlow imported ---")

MODEL_PATH = 'food_classifier.keras'

print("--- Loading Keras model... ---")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("--- Model loaded successfully! ---")
except Exception as e:
    print(f"--- Error loading model: {e} ---")
