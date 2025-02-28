# Common utlity functions

import os
import json
import logging
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.manifold import TSNE

def setup_logger(log_file="logs/pipeline.log", level=logging.INFO):
    """
    Set up a logger that writes to both the console and a file.
    
    Args:
        log_file (str): Path to the log file.
        level (int): Logging level (e.g., logging.INFO).
        
    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    # Ensure the log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Define log message format
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

def create_dir(directory):
    """
    Create a directory if it does not already exist.
    
    Args:
        directory (str): Directory path to create.
    """
    os.makedirs(directory, exist_ok=True)

def save_json(data, file_path):
    """
    Save a dictionary or list as a JSON file.
    
    Args:
        data (dict or list): Data to save.
        file_path (str): Path to the output JSON file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def load_json(file_path):
    """
    Load data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        The data loaded from the JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_dataframe(df, file_path):
    """
    Save a Pandas DataFrame to a CSV file.
    
    Args:
        df (pandas.DataFrame): DataFrame to save.
        file_path (str): Path to the output CSV file.
    """
    df.to_csv(file_path, index=False)

def load_yaml(file_path):
    """
    Load a YAML configuration file.
    
    Args:
        file_path (str): Path to the YAML file.
        
    Returns:
        dict: Parsed YAML configuration.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def plot_embeddings(embeddings, labels, output_file="embedding_plot.png", title="Embeddings Visualization"):
    """
    Visualize high-dimensional embeddings using t-SNE.
    
    Args:
        embeddings (np.array): Array of shape (n_samples, n_features) containing the embeddings.
        labels (list or np.array): Labels for each embedding (used for coloring).
        output_file (str): File path to save the plot.
        title (str): Title for the plot.
    """
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                          c=labels, cmap="viridis", s=10)
    plt.colorbar(scatter)
    plt.title(title)
    plt.savefig(output_file)
    plt.close()
    logging.info(f"Saved embedding visualization to {output_file}")
