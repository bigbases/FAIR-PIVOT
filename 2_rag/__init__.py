"""
Fairness Framework - A modular framework for generating balanced political datasets using RAG
"""

__version__ = "1.0.0"
__author__ = "Fairness Research Team"

# Avoid importing heavy or optional submodules at package import time
# to prevent circular or environment-specific import errors when only
# a subset (e.g., analyzers) is needed.
__all__ = []
