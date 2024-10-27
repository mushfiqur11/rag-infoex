import argparse
import os
from extract_from_pdf import process_pdf
from utils import (
    load_processed_files,
    load_faiss_index,
    save_faiss_index,
    save_processed_files,
    load_metadata,
    save_metadata,
)
from sentence_transformers import SentenceTransformer

def main(args):
    print(args)
    processed_files = load_processed_files(args.processed_file_path)

    # Check if metadata file exists and load it
    metadata_list = load_metadata(args.metadata_file)

    # Initialize the sentence transformer model
    model = SentenceTransformer(args.embedding_model)

    # Initialize FAISS index (or load it)
    sample_vector = model.encode(["sample text"])[0]  # Get vector dimensions
    vector_dim = len(sample_vector)
    index = load_faiss_index(vector_dim)

    # Iterate through all PDF files in the directory
    for pdf_file in os.listdir(args.knowledge_dir):
        if pdf_file.endswith(".pdf") and pdf_file not in processed_files:
            full_path = os.path.join(args.knowledge_dir, pdf_file)
            processed_files, metadata_list = process_pdf(full_path, index, metadata_list, processed_files)

    # Save the updated FAISS index
    save_faiss_index(index)

    # Save the updated metadata
    save_metadata(metadata_list, args.metadata_file)

    # Save the list of processed files
    save_processed_files(processed_files, args.processed_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--knowledge_dir', type=str, default='/scratch/mrahma45/nrl_rag_pipeline/rag-infoex/data')
    parser.add_argument('--processed_file_path', type=str, default='./processed_files.json')
    parser.add_argument('--metadata_file', type=str, default='./metadata.json')
    parser.add_argument('--embedding_model', type=str, default='all-MiniLM-L6-v2')

    args = parser.parse_args()
    main(args)