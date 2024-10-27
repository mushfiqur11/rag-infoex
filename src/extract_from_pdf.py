import PyPDF2
from sentence_transformers import SentenceTransformer
import os
import numpy as np
from transformers import AutoTokenizer
from typing import Union, List, Tuple, Dict

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text content from a PDF file.

    Args:
        pdf_path (str): The file path of the PDF to be processed.

    Returns:
        str: The extracted text from the PDF file as a single string.

    The function uses the PyPDF2 library to read the PDF file and extract the text
    content from each page. It concatenates the text from all the pages to return 
    a single string.
    """
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()  # Extract text from each page
    return text


def chunk_text(text: Union[List[str], str], tokenizer: AutoTokenizer, max_token_length: int) -> List:
    """
    Splits the extracted text into smaller chunks to ensure that each chunk
    does not exceed the specified maximum token length.

    Args:
        text (str): The complete extracted text to be chunked.
        tokenizer (AutoTokenizer): An autotokenizer model from huggingface
        max_token_length (int, optional): Maximum number of tokens allowed in each chunk. 
                                          Defaults to 512.

    Returns:
        List[str]: A list of text chunks where each chunk contains at most `max_token_length` tokens.

    The function first splits the text into paragraphs using double newline characters.
    It then aggregates paragraphs into chunks, ensuring that the total number of words
    in each chunk does not exceed the specified token limit.
    """
    paragraphs = text.split("\n")  # Split text into paragraphs based on double newlines
    chunks = []
    current_chunk = []
    current_length = 0
    # print(f"total number of paragraphs {len(paragraphs)}")

    # Iterate through paragraphs and add them to chunks based on token length
    for para in paragraphs:
        para_tokenized = tokenizer(para, padding=True, truncation=True, return_tensors="pt")
        para_length = len(para_tokenized[0])
        # print(f"para length {para_length}")
        if current_length + para_length <= max_token_length:
            current_chunk.append(para)
            current_length += para_length
        else:
            chunks.append(" ".join(current_chunk))  # Add the current chunk to chunks list
            current_chunk = [para]  # Start a new chunk with the current paragraph
            current_length = para_length

    if current_chunk:  # Add the final chunk if any paragraphs remain
        chunks.append(" ".join(current_chunk))

    return chunks


def vectorize_text_chunks(chunks: List[str], model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')) -> np.ndarray:
    """
    Converts text chunks into vector embeddings using a pre-trained SentenceTransformer model.

    Args:
        chunks (List[str]): A list of text chunks to be vectorized.
        model (SentenceTransformer, optional): A pre-trained SentenceTransformer model used for 
                                               generating embeddings. Defaults to 'all-MiniLM-L6-v2'.

    Returns:
        np.ndarray: A NumPy array containing the vector embeddings of the text chunks.

    The function encodes each text chunk into vector embeddings using the model’s `encode()` method,
    and returns the embeddings as a NumPy array.
    """
    vectors = model.encode(chunks)  # Encode the chunks into vector embeddings
    return vectors

def process_pdf(pdf_file: str, index, metadata_list, processed_files, llm_tokenizer, max_tokens=128) -> Tuple[Dict, List[Dict]]:
    """
    Processes a single PDF file by extracting text, chunking it into smaller parts,
    generating vector embeddings, updating the FAISS index, and storing metadata.

    Args:
        pdf_file (str): The file path of the PDF to be processed.
        index (faiss.Index): The FAISS index where vector embeddings will be added.
        metadata_list (List[Dict]): A list of metadata for storing information about the processed text chunks.
        processed_files (Dict): A dictionary that tracks which PDF files have already been processed.

    Returns:
        Tuple[Dict, List[Dict]]: 
            - Updated dictionary of processed files.
            - Updated list of metadata with information from the processed PDF.

    This function performs the following steps:
    1. Extracts text from the provided PDF file.
    2. Chunks the extracted text into smaller parts based on token length.
    3. Generates vector embeddings for each text chunk using the specified SentenceTransformer model.
    4. Adds the generated vectors to the FAISS index.
    5. Stores metadata for each text chunk, including the file name and chunk number.
    6. Marks the file as processed to avoid reprocessing in future runs.
    """
    print(f"Processing: {pdf_file}")
    
    # Extract the text from the PDF file
    pdf_text = extract_text_from_pdf(pdf_file)
    
    # Split the text into chunks for embedding
    chunks = chunk_text(pdf_text, llm_tokenizer, max_token_length=max_tokens)
    
    # Generate embeddings (vectors) for each chunk
    vectors = vectorize_text_chunks(chunks)
    
    # Add the vectors to the FAISS index
    vectors_np = np.array(vectors)
    index.add(vectors_np)  # FAISS indexing

    # Generate metadata for each chunk and append to metadata list
    metadata_list.extend([{
        "text": chunk,
        "file_name": os.path.basename(pdf_file),  # Store just the file name (not the full path)
        "chunk_id": i+1  # Assuming chunks correspond roughly to pages or sections
    } for i, chunk in enumerate(chunks)])

    # Mark the file as processed to avoid redundant processing
    processed_files[os.path.basename(pdf_file)] = True

    return processed_files, metadata_list


def search_faiss_index(query_vector, index, top_k=5):
    """
    Searches the FAISS index for the closest vectors to the provided query vector.

    Args:
        query_vector (np.ndarray): A single vector representing the query. This is the vector
                                   you want to search for in the FAISS index.
        index (faiss.Index): The FAISS index object that contains the stored vectors.
        top_k (int, optional): The number of nearest neighbors (top-k closest vectors) to return. 
                               Defaults to 5.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - indices (np.ndarray): An array of indices corresponding to the top-k closest vectors
                                    in the FAISS index.
            - distances (np.ndarray): An array of distances between the query vector and each of the
                                      top-k closest vectors. These distances represent the similarity
                                      (e.g., Euclidean distance) between the query vector and the 
                                      stored vectors.

    Example Usage:
        query_vector = np.random.rand(128)  # Assuming the vectors are 128-dimensional
        indices, distances = search_faiss_index(query_vector, index, top_k=10)
        print("Top 10 closest vectors' indices:", indices)
        print("Corresponding distances:", distances)

    This function uses FAISS's `search()` method to find the top-k closest vectors in the index
    to the input `query_vector`. It returns both the indices of the closest vectors and their
    corresponding distances. The `distances` array helps understand how similar the vectors are,
    while the `indices` array provides references to the vectors within the FAISS index.
    """
    distances, indices = index.search(np.array([query_vector]), top_k)
    return indices, distances

def search_with_window_retriever(query_vector, index, metadata, window_size=3, top_k=5):
    """
    Perform a FAISS search and expand context with window retrieval.
    """
    # Step 1: Perform FAISS search
    distances, indices = index.search(np.array([query_vector]), top_k)

    # Step 2: Expand context using a window retriever
    expanded_results = []
    # print(len(metadata))
    
    for idx in indices[0]:
        # print(idx)
        # relevant_text = metadata[idx]['text']
        file_name = metadata[idx]['file_name']
        chunk_id = metadata[idx]['chunk_id']

        # Get previous and next windows if they exist
        start_idx = max(0, idx - window_size)
        end_idx = min(len(metadata), idx + window_size + 1)

        context = " ".join([metadata[i]['text'] for i in range(start_idx, end_idx)])
        expanded_results.append({
            "context": context,
            "file_name": file_name,
            "chunk_id": chunk_id
        })

    return expanded_results