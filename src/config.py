"""
config.py
---------
This module loads configuration settings from the config.yaml file located in the project root.
"""

import os
from .utils import load_yaml

def get_config():
    """
    Load the configuration from the config.yaml file in the project's root folder.

    Returns:
        dict: The configuration settings as a dictionary.
    """
    # Determine the absolute path to the config.yaml file.
    # __file__ is the path to this file (config.py), so we move one level up to the project root.
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(root_dir, 'config.yaml')

    # Load and return the configuration using our utility function.
    return load_yaml(config_path)

if __name__ == "__main__":
    # For testing purposes, print the configuration.
    config = get_config()
    print(config)
