import requests
import sys
import os

# --- Configuration ---
URL = 'http://127.0.0.1:5000/predict'

def test_prediction(image_path):
    """Sends an image to the prediction API and prints the response."""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return

    print(f"Sending '{os.path.basename(image_path)}' to {URL} for prediction...")

    try:
        # Open the image file in binary mode
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            
            # Send the POST request
            response = requests.post(URL, files=files)

        # Print the results
        print("--- API Response ---")
        print(f"Status Code: {response.status_code}")
        try:
            print("JSON Response:", response.json())
        except requests.exceptions.JSONDecodeError:
            print("Response Content:", response.text)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while connecting to the API: {e}")

# --- Main Execution ---
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <path_to_image>")
        sys.exit(1)

    image_to_test = sys.argv[1]
    test_prediction(image_to_test)
