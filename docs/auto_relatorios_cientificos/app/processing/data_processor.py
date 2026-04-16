"""
Data processor module for AUTO-RELATÓRIOS CIENTÍFICOS
Handles data structuring and validation (updated for Sprint 2)
"""

from typing import Dict, List, Any
from ..config import REQUIRED_METADATA_FIELDS, CONFIG
from .analyzer import analyze_data, integrate_analysis_with_report

def validate_metadata(metadata: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate metadata has all required fields

    Args:
        metadata (dict): Metadata dictionary

    Returns:
        tuple: (is_valid, missing_fields)
    """
    missing_fields = []
    for field in REQUIRED_METADATA_FIELDS:
        if field not in metadata or not metadata[field]:
            missing_fields.append(field)

    return len(missing_fields) == 0, missing_fields

def structure_experiment_data(metadata: Dict[str, Any], data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Structure experiment data into a standardized format with analysis (Sprint 2)

    Args:
        metadata (dict): Experiment metadata
        data (list): Experimental data rows

    Returns:
        dict: Structured experiment data with analysis
    """
    structured_data = {
        "metadata": metadata,
        "data": data,
        "summary": {}
    }

    # Add basic statistics if data exists
    if data:
        structured_data["summary"]["data_points"] = len(data)
        # Count non-empty values for each column
        if data and isinstance(data[0], dict):
            columns = data[0].keys()
            structured_data["summary"]["columns"] = list(columns)

    # Perform advanced analysis if enabled
    if CONFIG.get("analysis", {}).get("enable_statistics", True):
        try:
            analysis_results = analyze_data(data, CONFIG)
            # Integrate analysis with structured data
            structured_data = integrate_analysis_with_report(structured_data, analysis_results)
        except Exception as e:
            # Continue without analysis if there's an error
            structured_data["analysis_error"] = str(e)

    return structured_data

def format_authors(authors_list: List[str]) -> str:
    """
    Format authors list into a string

    Args:
        authors_list (list): List of author names

    Returns:
        str: Formatted authors string
    """
    if not authors_list:
        return ""
    elif len(authors_list) == 1:
        return authors_list[0]
    elif len(authors_list) == 2:
        return f"{authors_list[0]} and {authors_list[1]}"
    else:
        return ", ".join(authors_list[:-1]) + f", and {authors_list[-1]}"