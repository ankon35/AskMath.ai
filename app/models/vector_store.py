import faiss
import numpy as np
from embedding import generate_embeddings_for_text
from config import GEMINI_API_KEY

def create_faiss_index(embedding_dimension):
    return faiss.IndexFlatL2(embedding_dimension)

def store_embeddings_in_faiss(embeddings, index, faiss_index_path="data/processed/faiss_index"):
    import os
    os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
    
    embeddings_np = np.array(embeddings).astype("float32")
    index.add(embeddings_np)
    faiss.write_index(index, faiss_index_path)
    print(f"✅ Embeddings stored successfully at: {faiss_index_path}")

# ----- Main Execution -----

# Your input text (can load from file or another source)
extracted_text = """
Sample extracted text from the PDF. This is a longer text to demonstrate chunking and embedding generation for multiple parts.
The more text you have, the more important it is to break it down into manageable pieces for API calls.
"""

# Step 1: Get embeddings from embedding.py
embeddings = generate_embeddings_for_text(extracted_text , api_key =GEMINI_API_KEY, model_id="models/embedding-001")

# Step 2: Create index and store if embeddings are valid
if embeddings and len(embeddings) > 0:
    embedding_dimension = len(embeddings[0])
    index = create_faiss_index(embedding_dimension)
    store_embeddings_in_faiss(embeddings, index)
else:
    print("❌ No embeddings were generated. Nothing to store.")
