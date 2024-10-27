import os
import json
import faiss
from typing import List
from glob import glob
import ntpath

# Default FAISS index file name
FAISS_INDEX_FILE = "faiss_index.idx"

def load_processed_files(file_path: str):
    """
    Loads a dictionary of processed files from a JSON file.

    Args:
        file_path (str): The path to the file storing the processed files.

    Returns:
        dict: A dictionary where the keys are file names and the values indicate 
              whether the file has been processed. If the file doesn't exist, 
              an empty dictionary is returned.

    The function checks if the file exists at the specified path. If it does, 
    the JSON content is loaded and returned as a dictionary. If not, it returns 
    an empty dictionary.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

def load_metadata(metadata_file: str):
    """
    Loads metadata from a JSON file.

    Args:
        metadata_file (str): The path to the file storing the metadata.

    Returns:
        List[dict]: A list of metadata dictionaries. If the file doesn't exist 
                    or contains invalid data, an empty list is returned.

    This function uses `load_processed_files()` to load the metadata stored in 
    the specified file. It checks if the loaded data is a valid list of metadata 
    and returns it. If invalid data is found, it returns an empty list.
    """
    metadata_list = load_processed_files(file_path=metadata_file)
    if isinstance(metadata_list, List):
        return metadata_list
    else:
        return []

def save_processed_files(processed_dict: dict, file_path: str) -> None:
    """
    Saves the dictionary of processed files to a JSON file.

    Args:
        processed_dict (dict): A dictionary where the keys are file names and 
                               the values indicate whether the file has been processed.
        file_path (str): The path to save the processed files dictionary.

    This function writes the `processed_dict` to the specified `file_path` 
    in JSON format.
    """
    with open(file_path, 'w') as f:
        json.dump(processed_dict, f, indent=4)

def save_metadata(metadata_list: dict, metadata_file: str) -> None:
    """
    Saves the metadata list to a JSON file.

    Args:
        metadata_list (List[dict]): A list of metadata dictionaries, where each 
                                    dictionary corresponds to a processed chunk of text.
        metadata_file (str): The path to save the metadata.

    This function writes the `metadata_list` to the specified `metadata_file` 
    in JSON format.
    """
    with open(metadata_file, 'w') as f:
        json.dump(metadata_list, f, indent=4)

def create_faiss_index(vector_dim):
    """
    Creates a new FAISS index with L2 distance.

    Args:
        vector_dim (int): The dimensionality of the vectors to be stored in the index.

    Returns:
        faiss.IndexFlatL2: A new FAISS index for L2 distance-based similarity searches.

    The function initializes a FAISS index using L2 distance, which is typically used 
    for finding similar vectors based on Euclidean distance.
    """
    index = faiss.IndexFlatL2(vector_dim)
    return index

def load_faiss_index(vector_dim, faiss_index_file: str = FAISS_INDEX_FILE, use_gpu: bool = False):
    """
    Loads an existing FAISS index from a file or creates a new one if it doesn't exist.

    Args:
        vector_dim (int): The dimensionality of the vectors to be stored in the index.
        faiss_index_file (str, optional): The file path of the FAISS index. Defaults to 'faiss_index.idx'.

    Returns:
        faiss.IndexFlatL2: A FAISS index loaded from the file or a newly created index if the file does not exist.

    The function first checks if the FAISS index file exists. If the file exists, 
    it loads the index from the file. Otherwise, it creates a new FAISS index 
    using `create_faiss_index()`.
    """
    if os.path.exists(faiss_index_file):
        index = faiss.read_index(faiss_index_file)
    else:
        index = create_faiss_index(vector_dim)
    
    if use_gpu:
        # Transfer the index to the GPU
        # res = faiss.StandardGpuResources()  # Initialize resources for GPU
        index = faiss.index_cpu_to_all_gpus(index=index)
        print(f"Transferred FAISS index to GPU.")

    return index

def save_faiss_index(index, faiss_index_file: str = FAISS_INDEX_FILE):
    """
    Saves the FAISS index to a file.

    Args:
        index (faiss.Index): The FAISS index to be saved.
        faiss_index_file (str, optional): The file path where the index will be saved. 
                                          Defaults to 'faiss_index.idx'.

    This function writes the FAISS index to the specified `faiss_index_file` using 
    the FAISS `write_index()` method. The saved index can later be loaded to 
    continue adding vectors or performing searches.
    """
    faiss.write_index(index, faiss_index_file)


def get_token(path):
    with open(path, 'r') as file:
        # Read the content of the file
        TOKEN = file.read().strip()  # Stripping any unnecessary whitespace/newlines
    return TOKEN

def update_arguments_with_config(parser, config_data, args):
    """
    Update the argument parser values based on the config file and parsed arguments.

    Args:
        parser (ArgumentParser): The argument parser object.
        config_data (dict): Dictionary of configuration values from the config file.
        args (Namespace): Parsed arguments from the command line.

    Returns:
        Namespace: Updated argument values with proper priority.
    """
    if not config_data:
        return parser.parse_args()
    # Convert parsed arguments to a dictionary
    parsed_args = vars(args)

    # Update argument parser defaults based on config file if command line arguments are not provided
    for key, value in config_data.items():
        if key not in parsed_args or parsed_args[key] is None:
            parser.set_defaults(**{key: value})

    # Re-parse the arguments with updated defaults
    final_args = parser.parse_args()
    
    return final_args

def load_config(config_file_path):
    """
    Load the configuration file in JSON format and return it as a dictionary.

    Args:
        config_file_path (str): Path to the configuration file.

    Returns:
        dict: Configuration values from the file.
    """
    if not os.path.exists(config_file_path):
        return False
    
    with open(config_file_path, 'r') as file:
        config_data = json.load(file)    
    return config_data

