"""
Presentation generator module for AUTO-RELATÓRIOS CIENTÍFICOS
Generates presentation scripts from structured data
"""

from typing import Dict, Any, List
from ..processing.data_processor import format_authors
from ..ai.integrator import enhance_section_text
from ..config import CONFIG

def fill_presentation_template(structured_data: Dict[str, Any], template_content: str) -> str:
    """
    Fill presentation template with structured data

    Args:
        structured_data (dict): Structured experiment data
        template_content (str): Template content with placeholders

    Returns:
        str: Filled presentation content
    """
    metadata = structured_data.get("metadata", {})
    data = structured_data.get("data", [])

    # Check if AI enhancement is enabled
    enable_ai_layer = CONFIG.get("ai", {}).get("enable_ai_layer", False)

    # Replace metadata placeholders
    presentation_content = template_content

    # Get original values
    title = metadata.get("title", "Untitled Experiment")
    authors = format_authors(metadata.get("authors", []))

    # Enhance with AI if enabled
    if enable_ai_layer:
        title = enhance_section_text(title, "Title of presentation", "scientific")

    # Replace placeholders with (possibly enhanced) values
    presentation_content = presentation_content.replace("{title}", title)
    presentation_content = presentation_content.replace("{authors}", authors)

    # Replace opening section
    opening = format_opening(metadata.get("title", ""), metadata.get("authors", []))
    if enable_ai_layer:
        opening = enhance_section_text(opening, "Opening of presentation", "scientific")
    presentation_content = presentation_content.replace("{opening}", opening)

    # Replace objective section
    objective = metadata.get("objective", "Not specified")
    if enable_ai_layer and objective != "Not specified":
        objective = enhance_section_text(objective, "Objective of presentation", "scientific")
    presentation_content = presentation_content.replace("{objective}", objective)

    # Replace methodology section
    methodology = format_methodology_for_presentation(metadata.get("methodology", ""))
    if enable_ai_layer and methodology != "Our methodology was designed to...":
        methodology = enhance_section_text(methodology, "Methodology of presentation", "scientific")
    presentation_content = presentation_content.replace("{methodology}", methodology)

    # Replace results section
    results = format_results_for_presentation(data)
    if enable_ai_layer and results != "Our experiments yielded no data.":
        results = enhance_section_text(results, "Results of presentation", "scientific")
    presentation_content = presentation_content.replace("{results}", results)

    # Replace interpretation section
    interpretation = format_interpretation_for_presentation(metadata.get("interpretation", metadata.get("analysis", "")))
    if enable_ai_layer and interpretation != "The data suggests that...":
        interpretation = enhance_section_text(interpretation, "Interpretation of presentation", "scientific")
    presentation_content = presentation_content.replace("{interpretation}", interpretation)

    # Replace conclusion section
    conclusion = format_conclusion_for_presentation(metadata.get("conclusion", "No conclusion provided."))
    if enable_ai_layer and conclusion != "No conclusion provided." and conclusion != "In conclusion, our work demonstrates...":
        conclusion = enhance_section_text(conclusion, "Conclusion of presentation", "scientific")
    presentation_content = presentation_content.replace("{conclusion}", conclusion)

    # Replace next steps section
    next_steps = format_next_steps_for_presentation(metadata.get("next_steps", []))
    if enable_ai_layer and next_steps != "Future work will focus on...":
        next_steps = enhance_section_text(next_steps, "Next steps of presentation", "scientific")
    presentation_content = presentation_content.replace("{next_steps}", next_steps)

    # Replace Q&A preparation section
    qa_preparation = metadata.get("qa_preparation", "Be prepared to answer questions about methodology and results.")
    if enable_ai_layer:
        qa_preparation = enhance_section_text(qa_preparation, "Q&A preparation of presentation", "scientific")
    presentation_content = presentation_content.replace("{qa_preparation}", qa_preparation)

    return presentation_content

def format_opening(title: str, authors: List[str]) -> str:
    """
    Format opening section for presentation

    Args:
        title (str): Presentation title
        authors (list): List of authors

    Returns:
        str: Formatted opening section
    """
    if not title and not authors:
        return "Good [morning/afternoon], today I'll be presenting our recent experimental work."

    if title and authors:
        return f"Good [morning/afternoon], today I'll be presenting '{title}' by {format_authors(authors)}."

    if title:
        return f"Good [morning/afternoon], today I'll be presenting '{title}'."

    return f"Good [morning/afternoon], today I'll be presenting our recent experimental work by {format_authors(authors)}."

def format_methodology_for_presentation(methodology: str) -> str:
    """
    Format methodology for presentation

    Args:
        methodology (str): Methodology description

    Returns:
        str: Formatted methodology
    """
    if not methodology:
        return "Our methodology was designed to..."

    return methodology

def format_results_for_presentation(data: List[Dict[str, Any]]) -> str:
    """
    Format results for presentation

    Args:
        data (list): List of data rows

    Returns:
        str: Formatted results
    """
    if not data:
        return "Our experiments yielded no data."

    return f"We collected {len(data)} data points during our experiments. Key findings include..."

def format_interpretation_for_presentation(interpretation: str) -> str:
    """
    Format interpretation for presentation

    Args:
        interpretation (str): Interpretation text

    Returns:
        str: Formatted interpretation
    """
    if not interpretation:
        return "The data suggests that..."

    return interpretation

def format_conclusion_for_presentation(conclusion: str) -> str:
    """
    Format conclusion for presentation

    Args:
        conclusion (str): Conclusion text

    Returns:
        str: Formatted conclusion
    """
    if not conclusion:
        return "In conclusion, our work demonstrates..."

    return conclusion

def format_next_steps_for_presentation(next_steps: List[str]) -> str:
    """
    Format next steps for presentation

    Args:
        next_steps (list): List of next steps

    Returns:
        str: Formatted next steps
    """
    if not next_steps:
        return "Future work will focus on..."

    steps = []
    for i, step in enumerate(next_steps, 1):
        steps.append(f"{i}. {step}")

    return "Next steps:\n" + "\n".join(steps)