"""
Statistical functions module for AUTO-RELATÓRIOS CIENTÍFICOS
Sprint 2 - Advanced data analysis
"""

import math
from typing import List, Dict, Any, Tuple, Union
from collections import Counter

def calculate_mean(values: List[float]) -> float:
    """
    Calculate arithmetic mean of a list of values

    Args:
        values (List[float]): List of numeric values

    Returns:
        float: Arithmetic mean
    """
    if not values:
        return 0.0
    return sum(values) / len(values)

def calculate_median(values: List[float]) -> float:
    """
    Calculate median of a list of values

    Args:
        values (List[float]): List of numeric values

    Returns:
        float: Median value
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)

    if n % 2 == 0:
        # Even number of elements - average of two middle values
        return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
    else:
        # Odd number of elements - middle value
        return sorted_values[n//2]

def calculate_standard_deviation(values: List[float]) -> float:
    """
    Calculate standard deviation of a list of values

    Args:
        values (List[float]): List of numeric values

    Returns:
        float: Standard deviation
    """
    if len(values) < 2:
        return 0.0

    mean = calculate_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)

def calculate_variance(values: List[float]) -> float:
    """
    Calculate variance of a list of values

    Args:
        values (List[float]): List of numeric values

    Returns:
        float: Variance
    """
    if len(values) < 2:
        return 0.0

    mean = calculate_mean(values)
    return sum((x - mean) ** 2 for x in values) / (len(values) - 1)

def calculate_range(values: List[float]) -> float:
    """
    Calculate range of a list of values

    Args:
        values (List[float]): List of numeric values

    Returns:
        float: Range (max - min)
    """
    if not values:
        return 0.0
    return max(values) - min(values)

def calculate_iqr(values: List[float]) -> float:
    """
    Calculate interquartile range (IQR) of a list of values

    Args:
        values (List[float]): List of numeric values

    Returns:
        float: Interquartile range
    """
    if len(values) < 4:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    q1_index = n // 4
    q3_index = 3 * n // 4

    q1 = sorted_values[q1_index]
    q3 = sorted_values[q3_index]

    return q3 - q1

def detect_outliers_tukey(values: List[float]) -> Tuple[List[int], List[float]]:
    """
    Detect outliers using Tukey's method (IQR method)

    Args:
        values (List[float]): List of numeric values

    Returns:
        Tuple[List[int], List[float]]: Indices and values of outliers
    """
    if len(values) < 4:
        return [], []

    q1 = calculate_median(sorted(values)[:len(values)//2])
    q3 = calculate_median(sorted(values)[len(values)//2:])
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_indices = []
    outlier_values = []

    for i, value in enumerate(values):
        if value < lower_bound or value > upper_bound:
            outlier_indices.append(i)
            outlier_values.append(value)

    return outlier_indices, outlier_values

def detect_outliers_zscore(values: List[float], threshold: float = 2.0) -> Tuple[List[int], List[float]]:
    """
    Detect outliers using Z-score method

    Args:
        values (List[float]): List of numeric values
        threshold (float): Z-score threshold for outlier detection (default: 2.0)

    Returns:
        Tuple[List[int], List[float]]: Indices and values of outliers
    """
    if len(values) < 3:
        return [], []

    mean = calculate_mean(values)
    std_dev = calculate_standard_deviation(values)

    if std_dev == 0:
        return [], []

    outlier_indices = []
    outlier_values = []

    for i, value in enumerate(values):
        z_score = abs((value - mean) / std_dev)
        if z_score > threshold:
            outlier_indices.append(i)
            outlier_values.append(value)

    return outlier_indices, outlier_values

def calculate_correlation(x_values: List[float], y_values: List[float]) -> float:
    """
    Calculate Pearson correlation coefficient between two lists of values

    Args:
        x_values (List[float]): First list of numeric values
        y_values (List[float]): Second list of numeric values

    Returns:
        float: Correlation coefficient (-1 to 1)
    """
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0

    x_mean = calculate_mean(x_values)
    y_mean = calculate_mean(y_values)

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))

    x_variance = sum((x - x_mean) ** 2 for x in x_values)
    y_variance = sum((y - y_mean) ** 2 for y in y_values)

    denominator = math.sqrt(x_variance * y_variance)

    if denominator == 0:
        return 0.0

    return numerator / denominator

def get_basic_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of values

    Args:
        values (List[float]): List of numeric values

    Returns:
        Dict[str, float]: Dictionary with statistical measures
    """
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std_dev": 0.0,
            "variance": 0.0,
            "min": 0.0,
            "max": 0.0,
            "range": 0.0
        }

    return {
        "count": len(values),
        "mean": calculate_mean(values),
        "median": calculate_median(values),
        "std_dev": calculate_standard_deviation(values),
        "variance": calculate_variance(values),
        "min": min(values),
        "max": max(values),
        "range": calculate_range(values)
    }