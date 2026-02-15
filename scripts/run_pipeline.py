#!/usr/bin/env python3
"""
EMG Analysis Pipeline Runner
===========================

This script executes the EMG analysis pipeline. It can be run with various options
from the command line.

Examples:
  1) Run all stages sequentially:
     python scripts/run_pipeline.py

  2) Run all stages in parallel (considering dependencies):
     python scripts/run_pipeline.py --parallel-pipeline

  3) Run specific stages:
     - By index:
       python scripts/run_pipeline.py --step 0
     - By name:
       python scripts/run_pipeline.py --step "01_build_dataset"
     - Multiple stages (comma-separated):
       python scripts/run_pipeline.py --step "0,1,2"
       python scripts/run_pipeline.py --step "01_build_dataset,02_emg_filtering"
     - Numeric range:
       python scripts/run_pipeline.py --step "0-2"

  4) Enable debug logs:
     python scripts/run_pipeline.py --debug
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import re
import sys
import time
import traceback
from pathlib import Path

from tqdm import tqdm

# Ensure the project root is importable.
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.stage_loader import STAGE_NAMES, get_stage_function
from src.utils import get_logger

DEFAULT_LOG_LEVEL = logging.INFO
logger: logging.Logger | None = None


def _configure_multiprocessing() -> None:
    """
    Keep Windows multiprocessing behavior consistent for users that run the
    pipeline on Windows.
    """
    if sys.platform != "win32":
        return
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method already set by another import; keep it.
        pass


def _initialize_logger(debug_mode: bool) -> logging.Logger:
    global logger
    level = logging.DEBUG if debug_mode else DEFAULT_LOG_LEVEL
    logger = get_logger("run_pipeline", level=level)
    if debug_mode:
        logger.info("Debug mode enabled. Log level set to DEBUG.")
    return logger


def run_stage(stage_name: str) -> None:
    """Execute a specific stage by name."""
    if logger is None:
        _initialize_logger(False)

    try:
        logger.info(f"Attempting to run stage: {stage_name}")

        stage_function = get_stage_function(stage_name)
        logger.info(f"Running run() function of {stage_name}...")

        result = stage_function()

        # Many stages return True/False; treat explicit False as failure.
        if result is False:
            raise RuntimeError(f"Stage returned False: {stage_name}")

        logger.info(f"Stage completed successfully: {stage_name}")

    except Exception as e:
        logger.error(f"Error occurred while running stage: {stage_name} - {str(e)}")
        logger.debug(traceback.format_exc())
        raise


def run_stages_with_optimal_parallelism() -> bool:
    """
    Execute stages in an optimized parallel manner.
    Runs independently executable stages in parallel considering dependencies.
    """
    if logger is None:
        _initialize_logger(False)

    # Updated dependencies for the current 3-stage pipeline.
    stage_dependencies: dict[str, list[str]] = {
        "01_build_dataset": [],
        "02_emg_filtering": ["01_build_dataset"],
        "03_post_processing": ["02_emg_filtering"],
    }

    completed_stages: set[str] = set()
    failed_stages: set[str] = set()

    def can_run_stage(stage_name: str) -> bool:
        return all(dep in completed_stages for dep in stage_dependencies.get(stage_name, []))

    def get_runnable_stages() -> list[str]:
        runnable: list[str] = []
        for stage in STAGE_NAMES:
            if stage not in completed_stages and stage not in failed_stages and can_run_stage(stage):
                runnable.append(stage)
        return runnable

    start_time = time.time()

    while len(completed_stages) + len(failed_stages) < len(STAGE_NAMES):
        runnable_stages = get_runnable_stages()

        if not runnable_stages:
            remaining_stages = set(STAGE_NAMES) - completed_stages - failed_stages
            if remaining_stages:
                logger.error(
                    f"Pipeline stuck. Remaining stages {remaining_stages} cannot run due to failed dependencies."
                )
                break
            break

        logger.info(f"Running stages in parallel: {runnable_stages}")

        max_parallel = min(len(runnable_stages), 4)
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

    total_time = time.time() - start_time

    logger.info(f"Pipeline execution completed in {total_time:.2f} seconds")
    logger.info(f"Completed stages: {len(completed_stages)}")
    logger.info(f"Failed stages: {len(failed_stages)}")

    if failed_stages:
        logger.error(f"Failed stages: {failed_stages}")
        return False

    logger.info("All stages completed successfully!")
    return True


def parse_step_argument(step_arg: str) -> list[str]:
    """
    Parse the step argument and return a list of stage names.

    Supports:
      - index: "0"
      - name: "01_build_dataset"
      - comma-separated: "0,1,2"
      - numeric range: "0-2"
    """
    if not step_arg:
        return []

    tokens = [token.strip() for token in step_arg.split(",")]
    resolved: list[str] = []

    for token in tokens:
        if re.fullmatch(r"\d+-\d+", token):
            start_idx, end_idx = map(int, token.split("-"))
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            if not (0 <= start_idx < len(STAGE_NAMES)) or not (0 <= end_idx < len(STAGE_NAMES)):
                raise ValueError(f"Range '{token}' is out of bounds 0-{len(STAGE_NAMES) - 1}.")
            resolved.extend(STAGE_NAMES[i] for i in range(start_idx, end_idx + 1))
            continue

        if token.isdigit():
            idx = int(token)
            if 0 <= idx < len(STAGE_NAMES):
                resolved.append(STAGE_NAMES[idx])
            else:
                raise ValueError(f"Stage index '{idx}' is out of bounds 0-{len(STAGE_NAMES) - 1}.")
        elif token in STAGE_NAMES:
            resolved.append(token)
        else:
            raise ValueError(
                f"Invalid stage specifier '{token}'. Must be index 0-{len(STAGE_NAMES) - 1} or valid stage name."
            )

    # De-duplicate while preserving order.
    seen: set[str] = set()
    ordered_unique: list[str] = []
    for stage in resolved:
        if stage not in seen:
            seen.add(stage)
            ordered_unique.append(stage)
    return ordered_unique


def main() -> int:
    _configure_multiprocessing()

    parser = argparse.ArgumentParser(description="EMG Analysis Pipeline - Main Execution Script")
    parser.add_argument(
        "--step",
        type=str,
        help=(
            "Run specific stages by number, name (e.g., '01_build_dataset'), "
            "comma-separated list ('0,1'), or numeric range ('0-2'). "
            "If not specified, all stages will be executed in order."
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--parallel-pipeline",
        action="store_true",
        help="Execute stages in an optimized parallel manner (considering dependencies).",
    )

    args = parser.parse_args()

    _initialize_logger(args.debug)

    logger.info("Starting EMG analysis pipeline...")
    logger.info(f"List of all available stages: {STAGE_NAMES}")

    if args.step:
        try:
            target_stage_names = parse_step_argument(args.step)
        except ValueError as e:
            logger.error(str(e))
            return 1

        logger.info(f"Running specified stages: {target_stage_names}")
        for stage_name in tqdm(target_stage_names, desc="Processing selected stages", unit="stage"):
            run_stage(stage_name)
        return 0

    if args.parallel_pipeline:
        logger.info("Running in optimized parallel pipeline mode...")
        success = run_stages_with_optimal_parallelism()
        return 0 if success else 1

    logger.info("Running all stages sequentially...")
    for i, stage_name in enumerate(tqdm(STAGE_NAMES, desc="Processing stages", unit="stage")):
        logger.info(f"--- Starting Stage {i:02d}: {stage_name} ---")
        run_stage(stage_name)
        logger.info(f"--- Completed Stage {i:02d}: {stage_name} ---")

    logger.info("All pipeline stages completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

