"""
Pipeline helpers for the EMG analysis project.

This package contains utilities for discovering and loading runnable stage
scripts from `scripts/`.
"""

from src.pipeline.stage_loader import STAGE_NAMES, get_stage_function, list_available_stages

__all__ = [
    "STAGE_NAMES",
    "get_stage_function",
    "list_available_stages",
]

