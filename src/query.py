from sentence_transformers import SentenceTransformer
import argparse
import os
import numpy as np
from utils import (
    load_faiss_index,
    load_metadata,
    get_token,
    load_config,
    update_arguments_with_config,
)
from extract_from_pdf import (
    vectorize_text_chunks,
    search_with_window_retriever
)
from llm_support import (
    formulate_prompt,
    get_model_and_tokenizer,
    generate_response
)
import torch

def main(args):
    """
    Main function to handle the retrieval of relevant information based on a query, using FAISS for vector search
    and a pre-trained language model for generating responses.

    Args:
        args (argparse.Namespace): The command-line arguments parsed by argparse, containing configurations like
                                   paths to models, embedding models, query, and more.

    Steps:
        1. Sets up the CUDA device if available.
        2. Retrieves the Hugging Face token for model access.
        3. Loads the sentence embedding model using SentenceTransformer.
        4. Loads metadata containing text chunks, file names, and other information from previously processed files.
        5. Vectorizes the user query for searching similar content in FAISS index.
        6. Uses FAISS to search for the top relevant results based on the query.
        7. Loads the LLM (language model) to generate a response based on retrieved information.
        8. Generates and prints the output sequence (model's response to the query).
    
    Returns:
        None
    """
    print(args)
    
    # Check GPU availability and set the device to CUDA
    print(torch.cuda.get_device_name(0))
    device = torch.device("cuda")
    print(device)

    # Retrieve Hugging Face token for model downloading
    HF_TOKEN = get_token(args.HF_TOKEN_PATH)

    # Load the embedding model using SentenceTransformer
    embedding_model = SentenceTransformer(args.embedding_model)

    # Load the metadata containing text, file names, etc.
    metadata = load_metadata(args.metadata_file)
    print(f"{len(metadata)} chunks available for retrieval")

    # If no query is provided, use a default example query
    if args.query is None:
        args.query = "What is the EIS machine?"

    # Vectorize the query using the sentence embedding model
    query_vector = vectorize_text_chunks([args.query], embedding_model)[0]

    # Get vector dimension (assuming query is a single vector)
    vector_dim = len(query_vector)

    # Load FAISS index for searching vectors
    faiss_index = load_faiss_index(vector_dim, faiss_index_file=args.faiss_index_file, use_gpu=False)

    # Perform a search in the FAISS index using the query vector
    expanded_results = search_with_window_retriever(query_vector=query_vector, index=faiss_index, metadata=metadata)

    # Load the language model and tokenizer from Hugging Face
    model, tokenizer = get_model_and_tokenizer(hf_model_path=args.hf_model_path, model_id=args.model_id, token=HF_TOKEN, redownload=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    # Formulate a prompt using the query and retrieved context (expanded results)
    prompt = formulate_prompt(args.query, expanded_results)
    
    # Generate a response from the model based on the prompt
    out_seq = generate_response(prompt=prompt, model=model, tokenizer=tokenizer, device=device, max_new_token=args.max_new_tokens)

    # Print the query and the model's generated response
    print(f"Query: {args.query}")
    print(f"Response: {out_seq}")

if __name__ == '__main__':
    """
    Entry point for the script when executed from the command line.

    Command-line Arguments:
        --knowledge_dir (str): Directory containing knowledge files.
        --vector_storage_path (str): Directory to store vector files.
        --processed_file_path (str): JSON file that keeps track of processed files.
        --metadata_file (str): JSON file that stores metadata (e.g., text chunks, file names).
        --faiss_index_file (str): FAISS index file path.
        --llm_model_path (str): Path to the directory where the LLM is stored.
        --embedding_model (str): The embedding model to be used for vectorization (default is 'all-MiniLM-L6-v2').
        --query (str): The query input from the user (default is None, meaning a sample query will be used).
        --hf_model_path (str): Directory to store Hugging Face models.
        --model_id (str): Hugging Face model ID to be used for generating responses (default is 'microsoft/Phi-3-mini-4k-instruct').
        --max_new_tokens (int): The maximum number of tokens to generate (default is 250).

    Returns:
        None
    """
    # Initialize the argument parser
    parser = argparse.ArgumentParser()

    # Define command-line arguments with default values
    parser.add_argument('--knowledge_dir', type=str, default='./data', help='Directory containing knowledge files.')
    parser.add_argument('--vector_storage_path', type=str, default='./vector_storage', help='Directory to store vector files.')
    parser.add_argument('--processed_file_path', type=str, default='processed_files.json', help='File for tracking processed PDFs.')
    parser.add_argument('--metadata_file', type=str, default='metadata.json', help='File for storing metadata like text chunks.')
    parser.add_argument('--faiss_index_file', type=str, default='faiss_index.idx', help='Path to the FAISS index file.')
    parser.add_argument('--llm_model_path', type=str, default='', help='Path to the large language model.')
    parser.add_argument('--embedding_model', type=str, default='all-MiniLM-L6-v2', help='Sentence embedding model to use.')
    parser.add_argument('--query', type=str, default=None, help='Query to search for, default is None.')
    parser.add_argument('--hf_model_path', type=str, default='./hf_models', help='Directory for Hugging Face models.')
    parser.add_argument('--model_id', type=str, default='microsoft/Phi-3-mini-4k-instruct', help='Model ID to use from Hugging Face.')
    parser.add_argument('--max_new_tokens', type=int, default=250, help='Maximum number of new tokens to generate in the response.')
    parser.add_argument('--config_path', type=str, default='./config.json', help='Path to the configuration file in JSON format.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Create directories for storing vectors and metadata if they don't exist
    if not os.path.exists(args.vector_storage_path):
        os.mkdir(args.vector_storage_path)

    # Load configuration from the specified config file
    config_file_path = args.config_path  # Path to the config file
    config_data = load_config(config_file_path)

    # Update arguments based on config file values
    args = update_arguments_with_config(parser, config_data, args)

    # Set paths for processed files and metadata
    args.processed_file_path = os.path.join(args.vector_storage_path, args.processed_file_path)
    args.metadata_file = os.path.join(args.vector_storage_path, args.metadata_file)
    args.faiss_index_file = os.path.join(args.vector_storage_path, args.faiss_index_file)

    # Execute the main function
    main(args)

