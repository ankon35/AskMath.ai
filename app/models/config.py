# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key and model ID from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID")
# GEMINI_Embedded_MODEL_ID = os.getenv("GEMINI_Embedded_MODEL_ID")

if not GEMINI_API_KEY or not GEMINI_MODEL_ID :
    raise ValueError("API key or model ID is missing in the .env file!")

print("API Key, Model ID and Embedded Model loaded successfully!")