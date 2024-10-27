import os
import json
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Directory containing PDF files
PDF_DIR = '/scratch/mrahma45/nrl_rag_pipeline/rag-infoex/data'

# File to store metadata of processed PDFs
PROCESSED_FILES = 'processed_files.json'

# FAISS index file
FAISS_INDEX_FILE = 'faiss_index.idx'

# JSON file to store metadata
METADATA_FILE = 'metadata.json'

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# def load_processed_files():
#     if os.path.exists(PROCESSED_FILES):
#         with open(PROCESSED_FILES, 'r') as f:
#             return json.load(f)
#     return {}

# def save_processed_files(processed_files):
#     with open(PROCESSED_FILES, 'w') as f:
#         json.dump(processed_files, f)

# def extract_text_from_pdf(pdf_path):
#     with open(pdf_path, 'rb') as file:
#         reader = PyPDF2.PdfReader(file)
#         text = ''
#         for page in range(len(reader.pages)):
#             text += reader.pages[page].extract_text()
#     return text

# def chunk_text(text, max_token_length=512):
#     paragraphs = text.split("\n\n")
#     chunks = []
#     current_chunk = []
#     current_length = 0

#     for para in paragraphs:
#         para_length = len(para.split())
#         if current_length + para_length <= max_token_length:
#             current_chunk.append(para)
#             current_length += para_length
#         else:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = [para]
#             current_length = para_length

#     if current_chunk:
#         chunks.append(" ".join(current_chunk))

#     return chunks

# def vectorize_text_chunks(chunks):
#     vectors = model.encode(chunks)
#     return vectors

# def create_faiss_index(vector_dim):
#     index = faiss.IndexFlatL2(vector_dim)
#     return index

# def load_faiss_index(vector_dim):
#     if os.path.exists(FAISS_INDEX_FILE):
#         index = faiss.read_index(FAISS_INDEX_FILE)
#     else:
#         index = create_faiss_index(vector_dim)
#     return index

# def save_faiss_index(index):
#     faiss.write_index(index, FAISS_INDEX_FILE)

# def process_pdf(pdf_file, index, metadata_list, processed_files):
#     print(f"Processing: {pdf_file}")
#     pdf_text = extract_text_from_pdf(pdf_file)
#     chunks = chunk_text(pdf_text)
#     vectors = vectorize_text_chunks(chunks)

#     # Add vectors to FAISS index
#     vectors_np = np.array(vectors)
#     index.add(vectors_np)

#     # Create metadata for each chunk
#     metadata_list.extend([{
#         "text": chunk,
#         "file_name": os.path.basename(pdf_file),
#         "page": i+1
#     } for i, chunk in enumerate(chunks)])

#     # Mark the file as processed
#     processed_files[os.path.basename(pdf_file)] = True

def process_pdfs_in_directory(directory):
    processed_files = load_processed_files()

    # Check if metadata file exists and load it
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            metadata_list = json.load(f)

    # Initialize FAISS index (or load it)
    sample_vector = model.encode(["sample text"])[0]  # Get vector dimensions
    vector_dim = len(sample_vector)
    index = load_faiss_index(vector_dim)

    # Iterate through all PDF files in the directory
    for pdf_file in os.listdir(directory):
        if pdf_file.endswith(".pdf") and pdf_file not in processed_files:
            full_path = os.path.join(directory, pdf_file)
            process_pdf(full_path, index, metadata_list, processed_files)

    # Save the updated FAISS index
    save_faiss_index(index)

    # Save the updated metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata_list, f)

    # Save the list of processed files
    save_processed_files(processed_files)

# Run the script
process_pdfs_in_directory(PDF_DIR)
