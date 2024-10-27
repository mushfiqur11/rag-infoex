import PyPDF2
from sentence_transformers import SentenceTransformer
import os
import numpy as np

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

def chunk_text(text, max_token_length=512):
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_length = len(para.split())
        if current_length + para_length <= max_token_length:
            current_chunk.append(para)
            current_length += para_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [para]
            current_length = para_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def vectorize_text_chunks(chunks, model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')):
    vectors = model.encode(chunks)
    return vectors

def process_pdf(pdf_file, index, metadata_list, processed_files):
    print(f"Processing: {pdf_file}")
    pdf_text = extract_text_from_pdf(pdf_file)
    chunks = chunk_text(pdf_text)
    vectors = vectorize_text_chunks(chunks)

    # Add vectors to FAISS index
    vectors_np = np.array(vectors)
    index.add(vectors_np)

    # Create metadata for each chunk
    metadata_list.extend([{
        "text": chunk,
        "file_name": os.path.basename(pdf_file),
        "page": i+1
    } for i, chunk in enumerate(chunks)])

    # Mark the file as processed
    processed_files[os.path.basename(pdf_file)] = True
    return processed_files, metadata_list