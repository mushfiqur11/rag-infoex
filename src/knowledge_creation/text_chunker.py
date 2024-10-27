import faiss
import numpy as np

def create_faiss_index(vectors):
    vectors_np = np.array(vectors)  # Convert to NumPy array
    vector_dim = vectors_np.shape[1]  # Dimensionality of vectors
    index = faiss.IndexFlatL2(vector_dim)  # L2 distance for similarity search
    index.add(vectors_np)  # Add vectors to the index
    return index

# Create the FAISS index
faiss_index = create_faiss_index(chunk_vectors)
