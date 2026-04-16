"""
Report generator module for AUTO-RELATÓRIOS CIENTÍFICOS
Generates scientific reports from structured data
"""

from typing import Dict, Any, List
from ..processing.data_processor import format_authors
from ..ai.integrator import enhance_section_text
from ..config import CONFIG

def fill_report_template(structured_data: Dict[str, Any], template_content: str) -> str:
    """
    Fill report template with structured data

    Args:
        structured_data (dict): Structured experiment data
        template_content (str): Template content with placeholders

    Returns:
        str: Filled report content
    """
    metadata = structured_data.get("metadata", {})
    data = structured_data.get("data", [])

    # Check if AI enhancement is enabled
    enable_ai_layer = CONFIG.get("ai", {}).get("enable_ai_layer", False)

    # Replace metadata placeholders
    report_content = template_content

    # Get original values
    title = metadata.get("title", "Untitled Experiment")
    authors = format_authors(metadata.get("authors", []))
    date = metadata.get("date", "Unknown")
    abstract = metadata.get("abstract", "No abstract provided.")
    introduction = metadata.get("introduction", "No introduction provided.")
    objective = metadata.get("objective", "Not specified")
    methodology = metadata.get("methodology", "Not specified")
    analysis = metadata.get("analysis", "No analysis provided.")
    conclusion = metadata.get("conclusion", "No conclusion provided.")
    references = metadata.get("references", "No references provided.")

    # Enhance with AI if enabled
    if enable_ai_layer:
        title = enhance_section_text(title, "Title of scientific report", "scientific")
        abstract = enhance_section_text(abstract, "Abstract of scientific report", "scientific")
        introduction = enhance_section_text(introduction, "Introduction of scientific report", "scientific")
        objective = enhance_section_text(objective, "Objective of scientific report", "scientific")
        methodology = enhance_section_text(methodology, "Methodology of scientific report", "scientific")
        analysis = enhance_section_text(analysis, "Analysis of scientific report", "scientific")
        conclusion = enhance_section_text(conclusion, "Conclusion of scientific report", "scientific")
        references = enhance_section_text(references, "References of scientific report", "scientific")

    # Replace placeholders with (possibly enhanced) values
    report_content = report_content.replace("{title}", title)
    report_content = report_content.replace("{authors}", authors)
    report_content = report_content.replace("{date}", date)
    report_content = report_content.replace("{abstract}", abstract)
    report_content = report_content.replace("{introduction}", introduction)
    report_content = report_content.replace("{objective}", objective)
    report_content = report_content.replace("{methodology}", methodology)
    report_content = report_content.replace("{analysis}", analysis)
    report_content = report_content.replace("{conclusion}", conclusion)
    report_content = report_content.replace("{references}", references)

    # Replace statistical analysis section
    statistical_analysis = metadata.get("statistical_analysis", "Statistical analysis not available.")
    if enable_ai_layer:
        statistical_analysis = enhance_section_text(statistical_analysis, "Statistical analysis section", "scientific")
    report_content = report_content.replace("{statistical_analysis}", statistical_analysis)

    # Replace outlier analysis section
    outlier_analysis = metadata.get("outlier_analysis", "Outlier analysis not available.")
    if enable_ai_layer:
        outlier_analysis = enhance_section_text(outlier_analysis, "Outlier analysis section", "scientific")
    report_content = report_content.replace("{outlier_analysis}", outlier_analysis)

    # Replace data visualization section
    data_visualization = metadata.get("data_visualization", "Data visualization not available.")
    if enable_ai_layer:
        data_visualization = enhance_section_text(data_visualization, "Data visualization section", "scientific")
    report_content = report_content.replace("{data_visualization}", data_visualization)

    # Replace data section
    data_section = format_data_section(data)
    report_content = report_content.replace("{data_section}", data_section)

    # Replace parameters section
    parameters_section = format_parameters_section(metadata.get("parameters", {}))
    report_content = report_content.replace("{parameters_section}", parameters_section)

    # Replace observations section
    observations_section = format_observations_section(metadata.get("observations", []))
    if enable_ai_layer and observations_section != "No observations recorded.":
        observations_section = enhance_section_text(observations_section, "Observations section", "scientific")
    report_content = report_content.replace("{observations_section}", observations_section)

    return report_content

def format_data_section(data: List[Dict[str, Any]]) -> str:
    """
    Format data section for the report

    Args:
        data (list): List of data rows

    Returns:
        str: Formatted data section
    """
    if not data:
        return "No experimental data available."

    # Create table header
    if isinstance(data[0], dict):
        headers = list(data[0].keys())
        table_lines = ["| " + " | ".join(headers) + " |"]
        table_lines.append("|" + "|".join(["---" for _ in headers]) + "|")

        # Add data rows
        for row in data:
            row_values = [str(row.get(header, "")) for header in headers]
            table_lines.append("| " + " | ".join(row_values) + " |")

        return "\n".join(table_lines)

    return "Data format not recognized."

def format_parameters_section(parameters: Dict[str, Any]) -> str:
    """
    Format parameters section for the report

    Args:
        parameters (dict): Parameters dictionary

    Returns:
        str: Formatted parameters section
    """
    if not parameters:
        return "No parameters specified."

    param_lines = []
    for key, value in parameters.items():
        param_lines.append(f"- **{key}**: {value}")

    return "\n".join(param_lines)

def format_observations_section(observations: List[str]) -> str:
    """
    Format observations section for the report

    Args:
        observations (list): List of observations

    Returns:
        str: Formatted observations section
    """
    if not observations:
        return "No observations recorded."

    obs_lines = []
    for i, observation in enumerate(observations, 1):
        obs_lines.append(f"{i}. {observation}")

    return "\n".join(obs_lines)