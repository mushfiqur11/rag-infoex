from llm_support import get_model_and_tokenizer
import argparse
from utils import (
    load_config,
    update_arguments_with_config,
    get_token
)

def main(args):
    """
    Main function responsible for downloading the specified LLM model and tokenizer.

    Args:
        args (argparse.Namespace): Parsed command-line arguments, including paths for 
                                   the model, token, and other configurations.
    
    Steps:
        1. Retrieves the Hugging Face token using the provided path.
        2. Downloads the model and tokenizer using Hugging Face API.
        3. Logs progress and completion messages.

    Returns:
        None
    """
    HF_TOKEN = get_token(args.HF_TOKEN_PATH)  # Get the Hugging Face API token
    print("Downloading LLM...")

    # Download the model and tokenizer using Hugging Face API with redownload option
    _, _ = get_model_and_tokenizer(
        hf_model_path=args.hf_model_path, 
        model_id=args.model_id, 
        token=HF_TOKEN, 
        redownload=True
    )
    
    print(f"Completed downloading {args.model_id} model.")

if __name__ == '__main__':
    # Initialize the argument parser
    parser = argparse.ArgumentParser()

    # Define command-line arguments with their default values
    parser.add_argument('--hf_model_path', type=str, default='./hf_models', help='Local path to save the Hugging Face models.')
    parser.add_argument('--model_id', type=str, default='microsoft/Phi-3-mini-4k-instruct', help='The ID of the model to download from Hugging Face.')
    parser.add_argument('--config_path', type=str, default='./config.json', help='Path to the configuration file in JSON format.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load configuration from the specified config file
    config_file_path = args.config_path  # Path to the config file
    config_data = load_config(config_file_path)

    # Update arguments based on config file values
    final_args = update_arguments_with_config(parser, config_data, args)

    # Execute the main function
    main(final_args)
