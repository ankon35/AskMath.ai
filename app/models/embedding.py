import requests
import os
import numpy as np
import time
from config import GEMINI_API_KEY
# Function to split large text into smaller chunks
def chunk_text(text, chunk_size=500):
    """
    Split the extracted text into smaller chunks to avoid exceeding token/character limits for embedding generation.

    :param text: The text content to chunk.
    :param chunk_size: The maximum size of each chunk.
    :return: A list of text chunks.
    """
    # Split the text into chunks of the specified size
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def generate_embeddings_for_chunk(chunk, api_key, model_id="models/embedding-001"):
    """
    Generate embeddings for a single text chunk using the Gemini API.

    :param chunk: A single chunk of text to generate embeddings for.
    :param api_key: Your Gemini API key.
    :param model_id: The ID of the Gemini embedding model to use.
    :return: A numpy array of embeddings for the chunk.
    """
    # Corrected endpoint for embedding content
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_id}:embedContent?key={api_key}"

    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_id,
        "content": {
            "parts": [{"text": chunk}]
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload)

        # Check for valid response and status
        if response.status_code == 200:
            data = response.json()
            # Correctly access the embedding values
            if "embedding" in data and "values" in data["embedding"]:
                return np.array(data["embedding"]["values"])  # Access 'values' key inside 'embedding'
            else:
                # Print the full response if embedding is missing for debugging
                print(f"Error: Missing 'embedding' or 'values' in the response: {data}")
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

def generate_embeddings_for_text(text, api_key, model_id="models/embedding-001"):
    """
    Split the text into chunks and generate embeddings for each chunk.

    :param text: The text content to generate embeddings for.
    :param api_key: Your Gemini API key.
    :param model_id: The ID of the Gemini embedding model to use.
    :return: A list of embeddings for each chunk.
    """
    # Step 1: Split the text into smaller chunks
    chunks = chunk_text(text)

    # Step 2: Generate embeddings for each chunk
    embeddings = []
    for chunk in chunks:
        embedding = generate_embeddings_for_chunk(chunk, api_key, model_id)
        if embedding is not None:
            embeddings.append(embedding)
        # Add a small delay between requests to avoid overwhelming the API
        time.sleep(1)  # 1-second delay to avoid rate limiting (you can adjust based on API's rate limit)

    # Return the list of embeddings for all chunks
    return embeddings

# Example usage: Generate embeddings for the extracted text
extracted_text = "Sample extracted text from the PDF. This is a longer text to demonstrate chunking and embedding generation for multiple parts. The more text you have, the more important it is to break it down into manageable pieces for API calls. Embeddings are numerical representations of text that capture semantic meaning. They are widely used in natural language processing tasks like search, recommendation, and classification. Generating good quality embeddings is crucial for the performance of these applications."  # Replace this with the actual extracted text

# Directly use the API key here (AGAIN, NOT RECOMMENDED FOR PRODUCTION)


GEMINI_MODEL_ID_DIRECT = "models/embedding-001"  # Using the publicly available embedding model

embeddings = generate_embeddings_for_text(extracted_text, GEMINI_API_KEY, GEMINI_MODEL_ID_DIRECT)

if embeddings:
    print("Generated Embeddings for Chunks:")
    for i, embedding in enumerate(embeddings[:5]):  # Print first 5 chunk embeddings
        print(f"Embedding {i+1} (first 10 values): {embedding[:10]}")
else:
    print("No embeddings were generated.")
