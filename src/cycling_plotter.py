"""Plotter for cycling data: Time vs Potential with integral as legend."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_time_potential_with_integral(df: pd.DataFrame, filename: str):
    """Plot Time vs Potential, add integral as legend."""
    plt.figure(figsize=(10, 6))
    plt.plot(df["tempo"], df["potencial"], label="Potential")

    # Calculate integral (area under curve, approximate energy proxy)
    integral = np.trapezoid(df["potencial"], df["tempo"])
    plt.title(f"{filename}: Time vs Potential\nIntegral: {integral:.2f}")
    plt.xlabel("Time")
    plt.ylabel("Potential")
    plt.legend()
    plt.grid(True)
    plt.show()