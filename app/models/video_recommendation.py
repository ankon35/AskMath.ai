import requests
import numpy as np
import faiss
import pandas as pd
from config import GEMINI_API_KEY
from embedding import generate_embeddings_for_chunk
from vector_store import create_faiss_index
import time
import base64

# Function to extract text from the image using Gemini API
def extract_text_from_image(image_path, api_key, model_id="models/gemini-2.5-flash"):
    """
    Extract text from an image using the Gemini API.
    
    :param image_path: Path to the image file.
    :param api_key: Gemini API key.
    :param model_id: Gemini model ID for text extraction.
    :return: Extracted text from the image.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_id}:generateContent?key={api_key}"

    headers = {
        "Content-Type": "application/json",
    }

    try:
        with open(image_path, "rb") as img_file:
            image_data_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": "Extract all text and mathematical expressions from this image."},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",  # Adjust mimeType based on your image type
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
            if data.get('candidates') and data['candidates'][0].get('content') and data['candidates'][0]['content'].get('parts'):
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

# Function to load the FAISS index
def create_faiss_index(faiss_index_path="data/processed/faiss_index"):
    """
    Load the FAISS index from disk.
    
    :param faiss_index_path: The path where the FAISS index is stored.
    :return: Loaded FAISS index.
    """
    index = faiss.read_index(faiss_index_path)
    return index

# Function to perform semantic search
def search_semantic(query_text, index, top_k=5):
    """
    Perform a semantic search by querying the FAISS index.
    
    :param query_text: The user query to search for.
    :param index: The FAISS index where embeddings are stored.
    :param top_k: The number of most similar results to return.
    :return: A list of the top K indices and distances.
    """
    # Generate the embedding for the query text
    query_embedding = generate_embeddings_for_chunk(query_text, api_key=GEMINI_API_KEY)  # Replace with your key
    
    if query_embedding is None:
        print("Error: Could not generate embedding for the query.")
        return None, None
    
    # Convert query embedding to numpy array
    query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)
    
    # Perform the search
    distances, indices = index.search(query_embedding, top_k)
    
    return distances, indices

# Function to fetch YouTube video and explanation from CSV
def get_video_recommendation(chapter, section, video_metadata_path="data/csv/Demo_Youtube_link.xlsx"):
    """
    Get the YouTube video recommendation based on the chapter and section.
    
    :param chapter: The matched chapter.
    :param section: The matched section.
    :param video_metadata_path: Path to the CSV file containing video metadata.
    :return: YouTube video title and link.
    """
    # Load video metadata from CSV
    video_df = pd.read_excel(video_metadata_path)
    
    # Find the row matching the chapter and section
    video_row = video_df[(video_df['অধ্যায়'] == chapter) | (video_df['অনুশীলনী'] == section)]

    
    if not video_row.empty:
        video_title = video_row.iloc[0]['YouTube link']
        return video_title
    else:
        return None

# Main function to integrate the process
def recommend_video_from_image(image_path):
    """
    Full flow to recommend video based on image input: extract text, perform semantic search, and fetch YouTube link.
    
    :param image_path: Path to the image file containing the math problem.
    """
    # Step 1: Extract text from the image using Gemini API
    extracted_text = extract_text_from_image(image_path, GEMINI_API_KEY)
    
    if extracted_text is None:
        print("❌ Text extraction failed. Exiting...")
        return
    
    print(f"Extracted Text: {extracted_text}")

    # Step 2: Generate embeddings for the extracted text
    embedding = generate_embeddings_for_chunk(extracted_text, GEMINI_API_KEY)

    if embedding is None:
        print("❌ Could not generate embedding.")
        return

    # Calculate the embedding dimension from the first generated embedding
    embedding_dimension = len(embedding)  # This will give the embedding dimension

    # Step 3: Load FAISS index
    index = create_faiss_index()  # Load FAISS index
    
    # Step 4: Perform semantic search using the extracted text embedding
    distances, indices = search_semantic(extracted_text, index, top_k=1)
    
    if indices is None or len(indices[0]) == 0:
        print("❌ No relevant chapter/section found.")
        return
    
    print(f"Top match index: {indices[0][0]}, Distance: {distances[0][0]}")

    # Step 5: Find the corresponding chapter and section from FAISS index
    # Assuming that the indices correspond to chapter and section info in the CSV metadata
    # This is a placeholder, you should implement a way to map FAISS indices to chapters/sections.
    chapter = "অধ্যায় ২"  # Retrieve this info based on FAISS results, here as an example
    section = "২.১ er ১. (ক)"  # Example section, would be extracted based on matching content
    
    # Step 6: Fetch the relevant YouTube video and explanation
    video_title = get_video_recommendation(chapter, section)
    
    if video_title:
        print(f"Recommended Video: {video_title}")
    else:
        print("❌ No video found for the selected chapter and section.")

# Example usage: Provide the path to the image
image_path = "data/images/test.png"
recommend_video_from_image(image_path)
