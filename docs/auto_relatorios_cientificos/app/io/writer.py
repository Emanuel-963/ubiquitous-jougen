"""
Writer module for AUTO-RELATÓRIOS CIENTÍFICOS
Handles writing output files
"""

import os
from typing import Dict, Any

def write_output(content: str, filename: str, output_dir: str = "output") -> str:
    """
    Write content to output file

    Args:
        content (str): Content to write
        filename (str): Output filename
        output_dir (str): Output directory path

    Returns:
        str: Path to the created file

    Raises:
        OSError: If file cannot be written
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        return filepath
    except OSError as e:
        raise OSError(f"Failed to write output file {filename}: {e}")

def write_multiple_outputs(contents: Dict[str, str], output_dir: str = "output") -> Dict[str, str]:
    """
    Write multiple output files

    Args:
        contents (dict): Dictionary mapping filenames to content
        output_dir (str): Output directory path

    Returns:
        dict: Dictionary mapping filenames to filepaths
    """
    filepaths = {}
    for filename, content in contents.items():
        filepaths[filename] = write_output(content, filename, output_dir)
    return filepaths