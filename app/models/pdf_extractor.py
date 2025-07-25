# pdf_extractor.py
import requests
import os
from config import GEMINI_API_KEY, GEMINI_MODEL_ID
import base64

def extract_text_from_pdf(pdf_path):
    """
    Extracts text content from the entire PDF file using the Gemini API.
    
    :param pdf_path: Path to the PDF file.
    :return: Extracted text content from the PDF.
    """
    # Ensure the PDF path is absolute to avoid FileNotFoundError issues
    absolute_pdf_path = os.path.abspath(pdf_path)

    if not os.path.exists(absolute_pdf_path):
        print(f"Error: PDF file not found at '{absolute_pdf_path}'")
        return None

    # Read the PDF file and encode it as base64
    try:
        with open(absolute_pdf_path, "rb") as pdf_file:
            pdf_data_base64 = base64.b64encode(pdf_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return None

    # Prepare the request URL for Gemini API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_ID}:generateContent?key={GEMINI_API_KEY}"

    headers = {
        "Content-Type": "application/json",
    }

    # Prepare payload to send the PDF content
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": "Extract all the content from this PDF."},
                    {
                        "inlineData": {
                            "mimeType": "application/pdf",  # Set MIME type for PDF
                            "data": pdf_data_base64
                        }
                    }
                ]
            }
        ]
    }

    # Send the request to the Gemini API
    try:
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            data = response.json()
            # Check if response contains the extracted content
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

# Example usage
pdf_path = "data/book/MathBook.pdf"
extracted_text = extract_text_from_pdf(pdf_path)

if extracted_text:
    print("\nExtracted PDF Content:")
    print(extracted_text[:500])  # Print first 500 characters of the extracted content for preview
