"""
Data analyzer module for AUTO-RELATÓRIOS CIENTÍFICOS
Sprint 2 - Advanced data analysis and outlier detection
"""

import yaml
import os
from typing import Dict, List, Any, Tuple, Optional
from ..utils.statistics import (
    calculate_mean, calculate_median, calculate_standard_deviation,
    detect_outliers_tukey, detect_outliers_zscore, get_basic_statistics
)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path (str): Path to configuration file

    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    if not os.path.exists(config_path):
        # Return default configuration if file doesn't exist
        return {
            "analysis": {
                "enable_statistics": True,
                "enable_outlier_detection": True,
                "outlier_method": "tukey",
                "confidence_level": 0.95,
                "enable_graphs": True,
                "graph_style": "ascii"
            }
        }

    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file) or {}

def extract_numeric_data(data: List[Dict[str, Any]], value_column: str = "value") -> List[float]:
    """
    Extract numeric values from data

    Args:
        data (List[Dict[str, Any]]): List of data rows
        value_column (str): Name of column containing numeric values

    Returns:
        List[float]: List of numeric values
    """
    numeric_values = []

    for row in data:
        if value_column in row:
            try:
                # Try to convert to float
                value = float(row[value_column])
                numeric_values.append(value)
            except (ValueError, TypeError):
                # Skip non-numeric values
                continue

    return numeric_values

def analyze_data(data: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform advanced analysis on experimental data

    Args:
        data (List[Dict[str, Any]]): Experimental data
        config (Dict[str, Any]): Configuration dictionary

    Returns:
        Dict[str, Any]: Analysis results
    """
    analysis_results = {
        "statistics": {},
        "outliers": {},
        "correlations": {},
        "summary": {}
    }

    # Check if analysis is enabled
    if not config.get("analysis", {}).get("enable_statistics", True):
        return analysis_results

    # Extract numeric data
    numeric_values = extract_numeric_data(data)

    if not numeric_values:
        analysis_results["summary"] = {
            "message": "No numeric data found for analysis",
            "data_points": 0
        }
        return analysis_results

    # Calculate basic statistics
    analysis_results["statistics"] = get_basic_statistics(numeric_values)
    analysis_results["summary"]["data_points"] = len(numeric_values)

    # Detect outliers if enabled
    if config.get("analysis", {}).get("enable_outlier_detection", True):
        outlier_method = config.get("analysis", {}).get("outlier_method", "tukey")

        if outlier_method == "tukey":
            outlier_indices, outlier_values = detect_outliers_tukey(numeric_values)
        elif outlier_method == "zscore":
            threshold = config.get("analysis", {}).get("outlier_threshold", 2.0)
            outlier_indices, outlier_values = detect_outliers_zscore(numeric_values, threshold)
        else:
            # Default to Tukey method
            outlier_indices, outlier_values = detect_outliers_tukey(numeric_values)

        analysis_results["outliers"] = {
            "method": outlier_method,
            "count": len(outlier_indices),
            "indices": outlier_indices,
            "values": outlier_values
        }

        # Add outlier summary
        if len(outlier_indices) > 0:
            analysis_results["summary"]["outliers_detected"] = True
            analysis_results["summary"]["outlier_percentage"] = (len(outlier_indices) / len(numeric_values)) * 100
        else:
            analysis_results["summary"]["outliers_detected"] = False

    return analysis_results

def format_statistics_report(statistics: Dict[str, float]) -> str:
    """
    Format statistics into a readable report

    Args:
        statistics (Dict[str, float]): Statistics dictionary

    Returns:
        str: Formatted statistics report
    """
    if not statistics or statistics.get("count", 0) == 0:
        return "No statistics available."

    report_lines = [
        "## Statistical Analysis",
        "",
        f"- **Data Points**: {statistics.get('count', 0)}",
        f"- **Mean**: {statistics.get('mean', 0):.4f}",
        f"- **Median**: {statistics.get('median', 0):.4f}",
        f"- **Standard Deviation**: {statistics.get('std_dev', 0):.4f}",
        f"- **Variance**: {statistics.get('variance', 0):.4f}",
        f"- **Minimum**: {statistics.get('min', 0):.4f}",
        f"- **Maximum**: {statistics.get('max', 0):.4f}",
        f"- **Range**: {statistics.get('range', 0):.4f}"
    ]

    return "\n".join(report_lines)

def format_outliers_report(outliers: Dict[str, Any]) -> str:
    """
    Format outliers into a readable report

    Args:
        outliers (Dict[str, Any]): Outliers dictionary

    Returns:
        str: Formatted outliers report
    """
    if not outliers or outliers.get("count", 0) == 0:
        return "No outliers detected."

    report_lines = [
        "## Outlier Detection",
        "",
        f"- **Method**: {outliers.get('method', 'unknown').title()}",
        f"- **Outliers Found**: {outliers.get('count', 0)}",
        f"- **Outlier Values**: {', '.join(map(str, outliers.get('values', [])))}"
    ]

    return "\n".join(report_lines)

def create_ascii_bar_chart(values: List[float], labels: Optional[List[str]] = None, width: int = 50) -> str:
    """
    Create ASCII bar chart from values

    Args:
        values (List[float]): Numeric values to plot
        labels (Optional[List[str]]): Labels for each bar
        width (int): Maximum width of the chart

    Returns:
        str: ASCII bar chart
    """
    if not values:
        return "No data to plot."

    max_value = max(values) if values else 1
    min_value = min(values) if values else 0

    # Handle case where all values are the same
    if max_value == min_value:
        max_value = min_value + 1

    chart_lines = ["## Data Visualization", ""]

    for i, value in enumerate(values):
        # Calculate bar length proportional to value
        bar_length = int(((value - min_value) / (max_value - min_value)) * width)
        bar = "█" * bar_length

        if labels and i < len(labels):
            chart_lines.append(f"{labels[i]}: {bar} ({value:.2f})")
        else:
            chart_lines.append(f"Item {i+1}: {bar} ({value:.2f})")

    return "\n".join(chart_lines)

def integrate_analysis_with_report(
    structured_data: Dict[str, Any],
    analysis_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Integrate analysis results with structured data

    Args:
        structured_data (Dict[str, Any]): Original structured data
        analysis_results (Dict[str, Any]): Analysis results

    Returns:
        Dict[str, Any]: Updated structured data with analysis
    """
    # Add analysis to structured data
    updated_data = structured_data.copy()
    updated_data["analysis"] = analysis_results

    # Add statistical summary to metadata if not present
    if "metadata" not in updated_data:
        updated_data["metadata"] = {}

    # Add analysis sections to metadata for template use
    if analysis_results.get("statistics"):
        stats_text = format_statistics_report(analysis_results["statistics"])
        updated_data["metadata"]["statistical_analysis"] = stats_text

    if analysis_results.get("outliers", {}).get("count", 0) > 0:
        outliers_text = format_outliers_report(analysis_results["outliers"])
        updated_data["metadata"]["outlier_analysis"] = outliers_text

    return updated_data