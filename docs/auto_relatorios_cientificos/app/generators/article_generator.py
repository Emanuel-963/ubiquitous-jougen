"""
Article generator module for AUTO-RELATÓRIOS CIENTÍFICOS
Generates short articles from structured data
"""

from typing import Dict, Any, List
from ..processing.data_processor import format_authors
from ..ai.integrator import enhance_section_text
from ..config import CONFIG

def fill_article_template(structured_data: Dict[str, Any], template_content: str) -> str:
    """
    Fill article template with structured data

    Args:
        structured_data (dict): Structured experiment data
        template_content (str): Template content with placeholders

    Returns:
        str: Filled article content
    """
    metadata = structured_data.get("metadata", {})
    data = structured_data.get("data", [])

    # Check if AI enhancement is enabled
    enable_ai_layer = CONFIG.get("ai", {}).get("enable_ai_layer", False)

    # Replace metadata placeholders
    article_content = template_content

    # Get original values
    title = metadata.get("title", "Untitled Experiment")
    authors = format_authors(metadata.get("authors", []))
    abstract = metadata.get("abstract", "No abstract provided.")
    introduction = metadata.get("introduction", "No introduction provided.")
    objective = metadata.get("objective", "Not specified")
    discussion = metadata.get("discussion", "No discussion provided.")
    conclusion = metadata.get("conclusion", "No conclusion provided.")
    references = metadata.get("references", "No references provided.")

    # Enhance with AI if enabled
    if enable_ai_layer:
        title = enhance_section_text(title, "Title of scientific article", "scientific")
        abstract = enhance_section_text(abstract, "Abstract of scientific article", "scientific")
        introduction = enhance_section_text(introduction, "Introduction of scientific article", "scientific")
        objective = enhance_section_text(objective, "Objective of scientific article", "scientific")
        discussion = enhance_section_text(discussion, "Discussion of scientific article", "scientific")
        conclusion = enhance_section_text(conclusion, "Conclusion of scientific article", "scientific")
        references = enhance_section_text(references, "References of scientific article", "scientific")

    # Replace placeholders with (possibly enhanced) values
    article_content = article_content.replace("{title}", title)
    article_content = article_content.replace("{authors}", authors)
    article_content = article_content.replace("{abstract}", abstract)
    article_content = article_content.replace("{introduction}", introduction)
    article_content = article_content.replace("{objective}", objective)
    article_content = article_content.replace("{discussion}", discussion)
    article_content = article_content.replace("{conclusion}", conclusion)
    article_content = article_content.replace("{references}", references)

    # Replace methodology section
    methodology_summary = format_methodology_summary(metadata.get("methodology", ""))
    if enable_ai_layer and methodology_summary != "Methodology not specified.":
        methodology_summary = enhance_section_text(methodology_summary, "Methodology summary", "scientific")
    article_content = article_content.replace("{methodology_summary}", methodology_summary)

    # Replace key results section
    key_results = format_key_results(data)
    if enable_ai_layer and key_results != "No experimental data available.":
        key_results = enhance_section_text(key_results, "Key results section", "scientific")
    article_content = article_content.replace("{key_results}", key_results)

    return article_content

def format_methodology_summary(methodology: str) -> str:
    """
    Format methodology summary for the article

    Args:
        methodology (str): Full methodology description

    Returns:
        str: Condensed methodology summary
    """
    if not methodology:
        return "Methodology not specified."

    # For now, just return the methodology as-is
    # Could implement summarization logic later
    return methodology

def format_key_results(data: List[Dict[str, Any]]) -> str:
    """
    Format key results for the article

    Args:
        data (list): List of data rows

    Returns:
        str: Formatted key results
    """
    if not data:
        return "No experimental data available."

    # Just show first few data points as examples
    summary_lines = [f"Total data points: {len(data)}"]

    if isinstance(data[0], dict):
        headers = list(data[0].keys())
        summary_lines.append(f"Data columns: {', '.join(headers)}")

        # Show first 3 rows as examples
        summary_lines.append("\nSample data:")
        for i, row in enumerate(data[:3]):
            row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
            summary_lines.append(f"  {i+1}. {row_str}")

    return "\n".join(summary_lines)

def format_conclusion(conclusion: str) -> str:
    """
    Format conclusion for the article

    Args:
        conclusion (str): Conclusion text

    Returns:
        str: Formatted conclusion
    """
    if not conclusion:
        return "No conclusion provided."

    return conclusion