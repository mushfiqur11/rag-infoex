import logging
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    TrainingArguments, 
    Trainer,
    AutoTokenizer
)
from huggingface_hub import snapshot_download
import re
import os

def get_model_and_tokenizer(hf_model_path: str, model_id: str, token: str, redownload: bool = False):
    """
    Downloads or loads a pre-trained Hugging Face model and tokenizer.

    Args:
        hf_model_path (str): The local directory where the model will be saved.
        model_id (str): The model ID from Hugging Face to download.
        token (str): Hugging Face authentication token.
        redownload (bool, optional): Whether to redownload the model if it already exists locally. Defaults to False.

    Returns:
        model (AutoModelForCausalLM): Loaded Hugging Face causal language model.
        tokenizer (AutoTokenizer): Loaded tokenizer corresponding to the model.

    The function checks if the model already exists locally in `hf_model_path`. If it does not exist, or if `redownload` is True,
    it downloads the model from Hugging Face using the specified token. The model and tokenizer are then returned for further use.
    """
    os.environ["HF_HOME"] = "./.cache"  # Set Hugging Face cache directory
    os.environ["TORCH_HOME"] = "./.cache"  # Set PyTorch cache directory

    model_path = os.path.join(hf_model_path, model_id)

    if (not os.path.exists(model_path)) or redownload:
        print(f"Trying to download model from {model_id}")
        snapshot_download(
            repo_id=model_id, 
            local_dir=model_path,
            token=token
        )

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def get_model_and_tokenizer_from_checkpoint(hf_checkpoint):
    """
    Loads a pre-trained model and tokenizer from a specified checkpoint.

    Args:
        hf_checkpoint (str): The path to the checkpoint of the model.

    Returns:
        model (AutoModelForCausalLM): Loaded model from the checkpoint.
        tokenizer (AutoTokenizer): Loaded tokenizer from the checkpoint.

    This function loads both the model and tokenizer directly from a pre-trained checkpoint.
    """
    model = AutoModelForCausalLM.from_pretrained(hf_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint)
    return model, tokenizer

def generate_response(prompt, model, tokenizer: AutoTokenizer, device, max_new_token=100):
    """
    Generates a response for a given input prompt using a pre-trained language model.

    Args:
        prompt (str): The input prompt for the model.
        model (AutoModelForCausalLM): The pre-trained language model.
        tokenizer (AutoTokenizer): The tokenizer for converting between text and tokens.
        device (torch.device): The device on which the model will run (e.g., CPU, GPU).
        max_new_token (int, optional): The maximum number of tokens to generate. Defaults to 100.

    Returns:
        str: The generated text output from the model.

    The function tokenizes the input prompt, generates a response based on the model configuration, and decodes the tokens to return the text output.
    """
    gen_config = {
        "temperature": 0.7,
        "top_p": 0.1,
        "repetition_penalty": 1.18,
        "top_k": 5,
        "do_sample": True,
        "max_new_tokens": max_new_token,
        "pad_token_id": tokenizer.eos_token_id
    }

    tokenized_input = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
    gen_tokens = model.generate(
        tokenized_input,
        **gen_config
    )
    input_seq = tokenizer.decode(tokenized_input[0])
    output_seq = tokenizer.decode(gen_tokens[0])

    if len(input_seq) < len(output_seq):
        output_seq = output_seq[len(input_seq):]
    
    return output_seq

def formulate_prompt(query: str, context: str):
    """
    Formulates a system-user interaction prompt for a RAG (Retrieval-Augmented Generation) model.

    Args:
        query (str): The user query.
        context (str): The retrieved context that is used to answer the query.

    Returns:
        List[dict]: A prompt structured as a list of dictionaries, where each dictionary contains the role
                    ('system' or 'user') and the corresponding content (query or system instruction).

    This function creates a chat-based prompt for the RAG model, where the system provides instructions and
    context, and the user provides the query.
    """
    prompt_dict = [
        {
            "role": "system",
            "content": f"You are a RAG model tasked to retrieve information. You will be provided with a user query followed by some retrieved bits. You need to answer the user query based on the retrieved context. Retrieved Context: {context}"
        },
        {
            "role": "user",
            "content": query
        }
    ]
    return prompt_dict