"""
Biomechanical Data Processing Stages

This package contains the processing stages for biomechanical data analysis.
"""

import importlib
from pathlib import Path

# Stage module names (cannot be imported directly due to numeric prefixes)
# Updated: Stage 02, 03, 04 unified into single 02_build_dataset
STAGE_MODULE_NAMES = [
    '01_build_dataset',         # NEW: Unified Stage 01
    '02_emg_filtering',
    '03_post_processing'
]

# Create stage function registry using dynamic imports
STAGE_FUNCTIONS = {}

for module_name in STAGE_MODULE_NAMES:
    try:
        # Import module dynamically
        module = importlib.import_module(f'.{module_name}', package=__name__)
        
        # Get the run function from the module
        if hasattr(module, 'run'):
            STAGE_FUNCTIONS[module_name] = module.run
        elif hasattr(module, 'main'):
            STAGE_FUNCTIONS[module_name] = module.main
        else:
            print(f"Warning: No run() or main() function found in {module_name}")
    except ImportError as e:
        print(f"Warning: Could not import {module_name}: {e}")

# Import some classes for backward compatibility
try:
    DatasetBuilder = importlib.import_module('.01_build_dataset', package=__name__).DatasetBuilder
except (ImportError, AttributeError):
    DatasetBuilder = None

__all__ = ['STAGE_FUNCTIONS', 'DatasetBuilder']
