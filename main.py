"""
# This script executes the EMG analysis pipeline. It can be run with various options from the command line.
#
# 1. Run all stages sequentially:
#    python main.py
#
# 2. Run all stages in parallel (considering dependencies):
#    python main.py --parallel-pipeline
#
# 3. Run specific stages:
#    You can specify one or more stages using the --step argument.
#
#    - Single stage (specified by number):
#      python main.py --step 0
#
#    - Single stage (specified by name):
#      python main.py --step "01_build_dataset"
#
#    - Multiple stages (comma-separated):
#      python main.py --step "0,1,2"
#      python main.py --step "01_build_dataset,02_emg_filtering"
#
#    - Multiple stages (specified by range):
#      python main.py --step "3-5"
#
# 4. Enable debug mode:
#    You can output detailed logs by adding the --debug flag.
#    python main.py --debug
#    python main.py --step 0 --debug
"""

import argparse
import importlib
import logging
import sys
import traceback
import os
from pathlib import Path
import subprocess
import concurrent.futures
import time
import re
from tqdm import tqdm

# Import stages package
from stages import STAGE_FUNCTIONS
from scr.utils import get_logger

# Changed to direct module loading approach
current_dir = Path(__file__).parent.resolve()

# Ensure the package root directory (parent of 'Module') is in sys.path
module_package_root = current_dir.parent
if str(module_package_root) not in sys.path:
    sys.path.insert(0, str(module_package_root))


# Default log level setting
DEFAULT_LOG_LEVEL = logging.INFO
logger = None  # Initialize later

# Stage definition (defined by file names within directory)
# Updated: Stage 01 Unified - Build comprehensive dataset from parquet
STAGE_MODULES = [
    "01_build_dataset",         # Stage 01: Build comprehensive dataset from parquet
    "02_emg_filtering",         # Stage 02: EMG signal processing (high-pass, demean, rectify, low-pass)
    "03_post_processing",       # Stage 03: Post-processing: configurable normalization and resampling
]

def _initialize_logger(debug_mode: bool):
    """Logger initialization function"""
    global logger
    level = logging.DEBUG if debug_mode else DEFAULT_LOG_LEVEL
    logger = get_logger("main", level=level)
    if debug_mode:
        logger.info("Debug mode enabled. Log level set to DEBUG.")
    return logger


def run_stage(stage_name: str):
    """Execute a specific stage"""
    if logger is None:  # If not yet initialized
        _initialize_logger(False)

    try:
        logger.info(f"Attempting to run stage: {stage_name}")

        # Get the stage function from STAGE_FUNCTIONS
        if stage_name not in STAGE_FUNCTIONS:
            logger.error(f"Stage not found: {stage_name}")
            logger.error(f"Available stages: {list(STAGE_FUNCTIONS.keys())}")
            raise ValueError(f"Unsupported stage: {stage_name}")

        stage_function = STAGE_FUNCTIONS[stage_name]
        
        logger.info(f"Running run() function of {stage_name}...")
        
        # Execute the stage function
        result = stage_function()

        # Many stages return True/False; treat explicit False as failure to stop the pipeline.
        if result is False:
            raise RuntimeError(f"Stage returned False: {stage_name}")
        
        logger.info(f"Stage completed successfully: {stage_name}")
        
    except Exception as e:
        logger.error(f"Error occurred while running stage: {stage_name} - {str(e)}")
        logger.debug(traceback.format_exc())
        raise

def run_stages_with_optimal_parallelism():
    """
    Execute stages in an optimized parallel manner.
    Run independently executable stages in parallel considering dependencies.
    """
    if logger is None:
        _initialize_logger(False)
    
    # Define stage dependencies (updated for unified Stage 02)
    stage_dependencies = {
        "01_build_dataset": [],
        "02_emg_filtering": ["01_build_dataset"],
        "03_post_processing": ["02_emg_filtering"],
    }
    
    completed_stages = set()
    failed_stages = set()
    
    def can_run_stage(stage_name):
        """Check if a stage can be run (dependencies satisfied)"""
        return all(dep in completed_stages for dep in stage_dependencies.get(stage_name, []))
    
    def get_runnable_stages():
        """Get stages that can be run now (dependencies satisfied)"""
        runnable = []
        for stage in STAGE_MODULES:
            if stage not in completed_stages and stage not in failed_stages and can_run_stage(stage):
                runnable.append(stage)
        return runnable
    
    start_time = time.time()
    
    while len(completed_stages) + len(failed_stages) < len(STAGE_MODULES):
        runnable_stages = get_runnable_stages()
        
        if not runnable_stages:
            # Check if we're stuck due to failed dependencies
            remaining_stages = set(STAGE_MODULES) - completed_stages - failed_stages
            if remaining_stages:
                logger.error(f"Pipeline stuck. Remaining stages {remaining_stages} cannot run due to failed dependencies.")
                break
            else:
                break  # All stages completed
        
        logger.info(f"Running stages in parallel: {runnable_stages}")
        
        # Run available stages in parallel (limited by CPU cores)
        max_parallel = min(len(runnable_stages), 4)  # Limit to 4 parallel stages to manage memory
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            future_to_stage = {executor.submit(run_stage, stage): stage for stage in runnable_stages}
            
            for future in concurrent.futures.as_completed(future_to_stage):
                stage = future_to_stage[future]
                try:
                    future.result()
                    completed_stages.add(stage)
                    logger.info(f"✓ Stage {stage} completed successfully")
                except Exception as e:
                    failed_stages.add(stage)
                    logger.error(f"✗ Stage {stage} failed: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"Pipeline execution completed in {total_time:.2f} seconds")
    logger.info(f"Completed stages: {len(completed_stages)}")
    logger.info(f"Failed stages: {len(failed_stages)}")
    
    if failed_stages:
        logger.error(f"Failed stages: {failed_stages}")
        return False
    else:
        logger.info("All stages completed successfully!")
        return True

def parse_step_argument(step_arg: str) -> list:
    """
    Parse the step argument and return a list of stage names.
    
    Args:
        step_arg: Step argument entered by the user
        
    Returns:
        List of stage names
        
    Raises:
        ValueError: Invalid step argument format
    """
    if not step_arg:
        return []
    
    # Process tokens separated by commas
    tokens = [token.strip() for token in step_arg.split(',')]
    resolved = []
    
    for token in tokens:
        # Detect numeric range pattern (e.g., 4-6)
        if re.fullmatch(r"\d+-\d+", token):
            start_idx, end_idx = map(int, token.split('-'))
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx  # allow reversed input
            if not (0 <= start_idx < len(STAGE_MODULES)) or not (0 <= end_idx < len(STAGE_MODULES)):
                raise ValueError(f"Range '{token}' is out of valid stage index bounds 0-{len(STAGE_MODULES)-1}.")
            resolved.extend(STAGE_MODULES[i] for i in range(start_idx, end_idx + 1))
        else:
            # Single stage index or name
            if token.isdigit():
                idx = int(token)
                if 0 <= idx < len(STAGE_MODULES):
                    resolved.append(STAGE_MODULES[idx])
                else:
                    raise ValueError(f"Stage index '{idx}' is out of bounds 0-{len(STAGE_MODULES)-1}.")
            elif token in STAGE_MODULES:
                resolved.append(token)
            else:
                raise ValueError(
                    f"Invalid stage specifier '{token}'. Must be index 0-{len(STAGE_MODULES)-1} or valid stage name."
                )

    # Remove duplicates while preserving order
    seen = set()
    ordered_unique = []
    for stage in resolved:
        if stage not in seen:
            seen.add(stage)
            ordered_unique.append(stage)
    return ordered_unique

def main():
    parser = argparse.ArgumentParser(description="EMG Analysis Pipeline - Main Execution Script")
    parser.add_argument(
        "--step",
        type=str,
        help=(
            "Run specific stages by number, name (e.g., '01_build_dataset'), "
            "comma-separated list ('0,1'), or numeric range ('0-2').\n"
            "If not specified, all stages will be executed in order."
        )
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging."
    )
    parser.add_argument(
        "--parallel-pipeline",
        action="store_true",
        help="Execute stages in an optimized parallel manner (considering dependencies)."
    )
    
    args = parser.parse_args()

    # Initialize logger
    _initialize_logger(args.debug)
    
    logger.info("Starting EMG analysis pipeline...")
    logger.info(f"List of all available stages: {STAGE_MODULES}")

    if args.step:
        try:
            target_stage_names = parse_step_argument(args.step)
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)

        logger.info(f"Running specified stages: {target_stage_names}")
        
        for stage_name in tqdm(target_stage_names, desc="Processing selected stages", unit="stage"):
            run_stage(stage_name)
    elif args.parallel_pipeline:
        logger.info("Running in optimized parallel pipeline mode...")
        success = run_stages_with_optimal_parallelism()
        if success:
            logger.info("All pipeline stages completed successfully.")
        else:
            logger.error("Some stages failed during pipeline execution.")
            sys.exit(1)
    else:
        logger.info("Running all stages sequentially...")
        for i, stage_name in enumerate(tqdm(STAGE_MODULES, desc="Processing stages", unit="stage")):
            logger.info(f"--- Starting Stage {i:02d}: {stage_name} ---")
            run_stage(stage_name)
            logger.info(f"--- Completed Stage {i:02d}: {stage_name} ---")
        logger.info("All pipeline stages completed successfully.")

if __name__ == "__main__":
    main()
