

# ocr.py
import requests
import os
import base64
from config import GEMINI_API_KEY, GEMINI_MODEL_ID

# Assuming config.py is in the same directory or accessible via PYTHONPATH
# from config import GEMINI_API_KEY, GEMINI_MODEL_ID

# Placeholder for API key and model ID if config.py is not available
# You should replace these with your actual API key and desired model ID
# IMPORTANT: You MUST replace the empty string below with your actual Gemini API Key.
# Get your API key from Google AI Studio: https://aistudio.google.com/app/apikey
 # Or "gemini-pro-vision" if you prefer

def extract_math_from_image(image_path):
    """
    Extracts text and mathematical expressions from an image using Gemini API.

    :param image_path: Path to the image file
    :return: Extracted text and mathematical expressions
    """
    # Ensure the image path is absolute to avoid FileNotFoundError issues
    # This makes the script more robust to where it's executed from.
    absolute_image_path = os.path.abspath(image_path)

    if not os.path.exists(absolute_image_path):
        print(f"Error: Image file not found at '{absolute_image_path}'")
        return None

    # Check if the API key is provided
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY is not set. Please provide your API key.")
        return None

    # Correct Gemini API endpoint for generateContent
    # The API key is passed as a query parameter
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_ID}:generateContent?key={GEMINI_API_KEY}"

    headers = {
        "Content-Type": "application/json",
    }

    try:
        with open(absolute_image_path, "rb") as img_file:
            # Encode image data to base64 for the API request
            image_data_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        # Correct payload structure for Gemini API image input
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": "Extract all text and mathematical expressions from this image."},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg", # Adjust mimeType based on your image type (e.g., image/png)
                                "data": image_data_base64
                            }
                        }
                    ]
                }
            ]
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            data = response.json()
            # Correctly parse the response for the generated text
            if data and data.get('candidates') and data['candidates'][0].get('content') and data['candidates'][0]['content'].get('parts'):
                extracted_text = data['candidates'][0]['content']['parts'][0].get('text', "")
                return extracted_text
            else:
                print(f"Error: Unexpected response structure from Gemini API: {data}")
                return None
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Network or API request error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Path where the image is stored.
# IMPORTANT: Make sure 'data/images/test.jpeg' is the correct relative path
# from where you are running this script, or provide an absolute path.
# For example, if 'ocr.py' is in 'app/' and 'test.jpeg' is in 'app/data/images/',
# then 'data/images/test.jpeg' is correct if you run the script from 'app/'.
# If you run it from the parent directory 'AskMath.ai/', then the path should be 'app/data/images/test.jpeg'.
image_path = "data/images/Equations.jpg"

# Example of using an absolute path if you know the exact location:
# image_path = "C:\\Users\\Tanjim\\Documents\\AskMath.ai\\AskMath.ai\\app\\data\\images\\test.jpeg"

extracted_content = extract_math_from_image(image_path)
if extracted_content:
    print("\nExtracted Content:")
    print(extracted_content)
