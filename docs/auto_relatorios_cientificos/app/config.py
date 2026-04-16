"""
Configuration module for AUTO-RELATÓRIOS CIENTÍFICOS
Supports both legacy configuration and YAML configuration (Sprint 2)
"""

import os
import yaml
from typing import Dict, Any

# Default configuration
DEFAULT_CONFIG = {
    "output": {
        "directory": "output",
        "formats": ["report", "article", "presentation"],
        "overwrite": True
    },
    "analysis": {
        "enable_statistics": True,
        "enable_outlier_detection": True,
        "outlier_method": "tukey",
        "confidence_level": 0.95,
        "enable_graphs": True,
        "graph_style": "ascii"
    },
    "data_processing": {
        "validate_headers": True,
        "required_columns": ["parameter", "value", "unit", "description"],
        "auto_detect_types": True
    },
    "templates": {
        "report": "templates/report_template.md",
        "article": "templates/article_template.md",
        "presentation": "templates/presentation_template.md"
    },
    "web": {
        "host": "localhost",
        "port": 5000,
        "debug": True,
        "max_upload_size_mb": 16
    },
    "ai": {
        "enable_ai_layer": False,
        "provider": "openai",
        "model": "gpt-4-turbo",
        "temperature": 0.3
    }
}

def load_yaml_config(config_path: str = "config.yaml") -> Dict[Any, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path (str): Path to configuration file

    Returns:
        Dict[Any, Any]: Configuration dictionary
    """
    if not os.path.exists(config_path):
        return DEFAULT_CONFIG

    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            yaml_config = yaml.safe_load(file)
            if yaml_config:
                # Merge with default config
                merged_config = DEFAULT_CONFIG.copy()
                for key, value in yaml_config.items():
                    if isinstance(value, dict) and key in merged_config:
                        merged_config[key].update(value)
                    else:
                        merged_config[key] = value
                return merged_config
            return DEFAULT_CONFIG
    except Exception:
        # Return default config if YAML parsing fails
        return DEFAULT_CONFIG

# Load configuration
CONFIG = load_yaml_config()

# Legacy constants for backward compatibility
OUTPUT_DIR = CONFIG["output"]["directory"]
DATA_DIR = "data"
EXAMPLES_DIR = "data/examples"

# Template directory
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# Supported output formats
OUTPUT_FORMATS = CONFIG["output"]["formats"]

# Default template names
TEMPLATE_FILES = CONFIG["templates"]

# Required metadata fields
REQUIRED_METADATA_FIELDS = [
    "title",
    "authors",
    "objective",
    "methodology"
]

# Default CSV headers for validation
DEFAULT_CSV_HEADERS = CONFIG["data_processing"]["required_columns"]