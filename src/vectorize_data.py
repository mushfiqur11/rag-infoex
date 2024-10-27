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
from transformers import AutoTokenizer

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
    index = load_faiss_index(vector_dim, faiss_index_file=args.faiss_index_file)

    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_path)

    # Iterate through all PDF files in the directory
    for pdf_file in os.listdir(args.knowledge_dir):
        if pdf_file.endswith(".pdf") and pdf_file not in processed_files:
            full_path = os.path.join(args.knowledge_dir, pdf_file)
            processed_files, metadata_list = process_pdf(full_path, index, metadata_list, processed_files, llm_tokenizer, args.max_tokens)

    # Save the updated FAISS index
    save_faiss_index(index, faiss_index_file=args.faiss_index_file)

    # Save the updated metadata
    save_metadata(metadata_list, args.metadata_file)

    # Save the list of processed files
    save_processed_files(processed_files, args.processed_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--current_dir', type=str, default='src/knowledge_creation')
    parser.add_argument('--knowledge_dir', type=str, default='./data')
    parser.add_argument('--vector_storage_path', type=str, default='./vector_storage')
    parser.add_argument('--processed_file_path', type=str, default='processed_files.json')
    parser.add_argument('--metadata_file', type=str, default='metadata.json')
    parser.add_argument('--faiss_index_file', type=str, default='faiss_index.idx')
    parser.add_argument('--embedding_model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--max_tokens', type=int, default=128)
    parser.add_argument('--llm_path', type=str, default='/scratch/mrahma45/dialect_toxicity_llm_judge/hf_models/microsoft/Phi-3-mini-4k-instruct')

    args = parser.parse_args()
    if not os.path.exists(args.vector_storage_path):
        os.mkdir(args.vector_storage_path)
    args.processed_file_path = os.path.join(args.vector_storage_path, args.processed_file_path)
    args.metadata_file = os.path.join(args.vector_storage_path, args.metadata_file)
    args.faiss_index_file = os.path.join(args.vector_storage_path, args.faiss_index_file)
    main(args)