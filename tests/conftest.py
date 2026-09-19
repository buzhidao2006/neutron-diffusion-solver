"""Pytest configuration shared by plotting tests."""

import matplotlib


# Figures are asserted structurally; tests must not require a desktop Tk install.
matplotlib.use("Agg")
