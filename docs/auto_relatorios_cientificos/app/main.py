#!/usr/bin/env python3
"""
Main entry point for AUTO-RELATÓRIOS CIENTÍFICOS
Scientific reporting system for automatic generation of reports, articles, and presentations
"""

import sys
import os
import argparse

# Add the project root directory to the path so we can import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.io.reader import read_metadata, read_data
from app.io.writer import write_multiple_outputs
from app.processing.data_processor import validate_metadata, structure_experiment_data
from app.generators.report_generator import fill_report_template
from app.generators.article_generator import fill_article_template
from app.generators.presentation_generator import fill_presentation_template
from app.utils.validator import validate_metadata_fields
from app.config import OUTPUT_DIR, TEMPLATES_DIR, TEMPLATE_FILES


def load_template(template_name: str) -> str:
    """Load template content from file"""
    template_file = TEMPLATE_FILES.get(template_name)
    if not template_file:
        raise ValueError(f"Unknown template: {template_name}")

    # Try relative to the script location first
    template_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates", template_file)

    # Try absolute path if relative doesn't work
    if not os.path.exists(template_path):
        template_path = os.path.join(TEMPLATES_DIR, template_file)

    with open(template_path, 'r', encoding='utf-8') as file:
        return file.read()


def generate_reports(metadata_path: str, data_path: str, output_dir: str = OUTPUT_DIR) -> dict:
    """
    Generate all report types from metadata and data files

    Args:
        metadata_path (str): Path to metadata JSON file
        data_path (str): Path to data CSV file
        output_dir (str): Output directory path

    Returns:
        dict: Dictionary mapping filenames to filepaths
    """
    # Read input files
    metadata = read_metadata(metadata_path)
    data = read_data(data_path)

    # Validate metadata
    is_valid, errors = validate_metadata_fields(metadata)
    if not is_valid:
        raise ValueError(f"Invalid metadata: {'; '.join(errors)}")

    # Structure data
    structured_data = structure_experiment_data(metadata, data)

    # Load templates
    report_template = load_template("report")
    article_template = load_template("article")
    presentation_template = load_template("presentation")

    # Generate content
    report_content = fill_report_template(structured_data, report_template)
    article_content = fill_article_template(structured_data, article_template)
    presentation_content = fill_presentation_template(structured_data, presentation_template)

    # Prepare outputs
    outputs = {
        "relatorio.md": report_content,
        "artigo.md": article_content,
        "apresentacao.md": presentation_content
    }

    # Write outputs
    filepaths = write_multiple_outputs(outputs, output_dir)
    return filepaths


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AUTO-RELATÓRIOS CIENTÍFICOS - Generate scientific reports automatically")
    parser.add_argument("metadata", help="Path to metadata JSON file")
    parser.add_argument("data", help="Path to data CSV file")
    parser.add_argument("-o", "--output", default=OUTPUT_DIR, help="Output directory (default: output)")

    args = parser.parse_args()

    try:
        print("Generating scientific reports...")
        filepaths = generate_reports(args.metadata, args.data, args.output)

        print("Reports generated successfully:")
        for filename, filepath in filepaths.items():
            print(f"  - {filepath}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()