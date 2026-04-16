"""
Data reader module for AUTO-RELATÓRIOS CIENTÍFICOS
Handles reading metadata and data files
"""

import json
import csv
import os
from typing import Dict, List, Any

def read_metadata(filepath: str) -> Dict[str, Any]:
    """
    Read metadata from JSON file

    Args:
        filepath (str): Path to the JSON file

    Returns:
        dict: Metadata dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metadata file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as file:
        try:
            metadata = json.load(file)
            return metadata
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in metadata file: {e}", e.doc, e.pos)

def read_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Read experimental data from CSV file

    Args:
        filepath (str): Path to the CSV file

    Returns:
        list: List of dictionaries representing CSV rows

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If CSV is malformed
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        try:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
            return data
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

def validate_input_files(metadata_path: str, data_path: str) -> bool:
    """
    Validate that input files exist and are readable

    Args:
        metadata_path (str): Path to metadata file
        data_path (str): Path to data file

    Returns:
        bool: True if both files are valid
    """
    try:
        read_metadata(metadata_path)
        read_data(data_path)
        return True
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return False