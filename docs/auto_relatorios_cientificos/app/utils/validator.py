"""
Validator module for AUTO-RELATÓRIOS CIENTÍFICOS
Handles input validation
"""

from typing import Dict, Any, List

def validate_metadata_fields(metadata: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate that metadata contains all required fields

    Args:
        metadata (dict): Metadata dictionary

    Returns:
        tuple: (is_valid, list_of_errors)
    """
    required_fields = [
        "title",
        "authors",
        "objective",
        "methodology"
    ]

    errors = []

    for field in required_fields:
        if field not in metadata:
            errors.append(f"Missing required field: {field}")
        elif not metadata[field]:
            errors.append(f"Empty required field: {field}")

    # Special validation for authors
    if "authors" in metadata and not isinstance(metadata["authors"], list):
        errors.append("Authors field must be a list")

    return len(errors) == 0, errors

def validate_csv_headers(headers: List[str]) -> tuple[bool, List[str]]:
    """
    Validate CSV headers

    Args:
        headers (list): List of CSV headers

    Returns:
        tuple: (is_valid, list_of_warnings)
    """
    warnings = []

    if not headers:
        warnings.append("No headers found in CSV file")
        return False, warnings

    # Check for common expected headers
    expected_headers = {"parameter", "value", "unit", "description"}
    found_headers = set(h.lower() for h in headers)

    missing_headers = expected_headers - found_headers
    if missing_headers:
        warnings.append(f"Missing expected headers: {missing_headers}")

    return len(warnings) == 0, warnings