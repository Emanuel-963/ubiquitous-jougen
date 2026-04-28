"""pytest configuration — applied before any test module is collected."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for all tests (no Tk needed)
