import os
import json
import faiss
from typing import List

def load_processed_files(file_path: str):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

def load_metadata(metadata_file: str):
    metadata_list = load_processed_files(file_path=metadata_file)
    if isinstance(metadata_list, List):
        return metadata_list
    else:
        return []

def save_processed_files(processed_dict: dict, file_path: str) -> None:
    with open(file_path, 'w') as f:
        json.dump(processed_dict, f)

def save_metadata(metadata_list: dict, metadata_file: str) -> None:
    with open(metadata_file, 'w') as f:
        json.dump(metadata_list, f)

def create_faiss_index(vector_dim):
    index = faiss.IndexFlatL2(vector_dim)
    return index

def load_faiss_index(vector_dim, FAISS_INDEX_FILE: str = "faiss_index.idx"):
    if os.path.exists(FAISS_INDEX_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)
    else:
        index = create_faiss_index(vector_dim)
    return index

def save_faiss_index(index, FAISS_INDEX_FILE: str = "faiss_index.idx"):
    faiss.write_index(index, FAISS_INDEX_FILE)

