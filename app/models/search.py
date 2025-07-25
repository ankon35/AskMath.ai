import faiss
import numpy as np
from embedding import generate_embeddings_for_chunk  # Assuming this function is in embedding.py
from config import GEMINI_API_KEY

def load_faiss_index(faiss_index_path="data/processed/faiss_index"):
    """
    Load the FAISS index from disk.
    
    :param faiss_index_path: The path where the FAISS index is stored.
    :return: Loaded FAISS index.
    """
    index = faiss.read_index(faiss_index_path)
    return index

def search_semantic(embedding, index, top_k=5):
    """
    Perform a semantic search by querying the FAISS index.
    
    :param embedding: The query embedding to search for.
    :param index: The FAISS index where embeddings are stored.
    :param top_k: The number of most similar results to return.
    :return: A list of the top K indices and distances.
    """
    # Convert query embedding to numpy array if it's not already
    query_embedding = np.array(embedding).astype("float32").reshape(1, -1)
    
    # Perform the search
    distances, indices = index.search(query_embedding, top_k)
    
    return distances, indices

# Example usage: Querying the FAISS index
query_text = "Find the most relevant chapter on Real Number."
query_embedding = generate_embeddings_for_chunk(query_text, GEMINI_API_KEY)  # Get embedding for query

# Load FAISS index from file
index = load_faiss_index()

# Perform search and get top 5 most similar embeddings
distances, indices = search_semantic(query_embedding, index, top_k=5)

print(f"Top 5 search results (indices): {indices}")
print(f"Top 5 distances (similarity): {distances}")
