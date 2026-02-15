#!/usr/bin/env python3
"""
Stage 03 entrypoint.

Implementation lives in `src/stages/stage03_post_processing.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is importable when executed directly.
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_helpers import get_output_path, load_config_yaml
from src.stages.stage03_post_processing import _resolve_stage02_input_file, logger, process_post_processing_steps
from src.utils import write_summary


def run():
    """Run function for main.py compatibility."""
    input_dir = get_output_path("02_processed", "")
    output_dir = get_output_path("03_post_processed", "")
    config_path = "config.yaml"

    logger.info("Starting Stage 03: Post Processing")

    try:
        config = load_config_yaml(config_path)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return False

    processed_name = config.get("pipeline_files", {}).get("stage02_processed_emg") or "processed_emg_data.parquet"
    sig_cfg = config.get("signal_processing", {}) or {}
    selected_option = sig_cfg.get("selected_option")
    try:
        input_file = _resolve_stage02_input_file(
            input_dir=input_dir,
            processed_name=processed_name,
            selected_option=selected_option,
        )
    except Exception as e:
        logger.error(f"Failed to resolve Stage02 input file: {e}")
        return False
    if input_file is None:
        logger.error(f"No processed EMG data found. Expected '{processed_name}' in {input_dir}")
        return False

    success = process_post_processing_steps(input_file, output_dir, config)

    if success:
        logger.info("Stage 03: Post Processing completed successfully")

        norm_cfg = config.get("normalization", {}) or {}
        execution_order = norm_cfg.get("execution_order", []) or []
        definitions = norm_cfg.get("definitions", {}) or {}
        step_descriptions: list[str] = []
        for step_id in execution_order:
            step_cfg = definitions.get(step_id, {}) if isinstance(definitions, dict) else {}
            method = step_cfg.get("method", "?") if isinstance(step_cfg, dict) else "?"
            if method == "baseline":
                window_ms = step_cfg.get("baseline_window_ms") if isinstance(step_cfg, dict) else None
                if window_ms is not None:
                    step_descriptions.append(f"{step_id}:{method}(window_ms={window_ms})")
                else:
                    step_descriptions.append(f"{step_id}:{method}")
            else:
                grouping = step_cfg.get("grouping_columns") if isinstance(step_cfg, dict) else None
                if grouping is not None:
                    step_descriptions.append(f"{step_id}:{method}(grouping={grouping})")
                else:
                    step_descriptions.append(f"{step_id}:{method}")

        summary_stats = {
            "status": "Success",
            "input_file": str(input_file),
            "output_directory": str(output_dir),
            "normalization_execution_order": execution_order,
            "normalization_steps": " -> ".join(step_descriptions) if step_descriptions else "disabled",
            "final_output": config.get("pipeline_files", {}).get("stage03_normalized") or "normalized_data.parquet",
        }

        summary_file = output_dir / "post_processing_summary.txt"
        write_summary("Stage 03: Post Processing", summary_stats, summary_file)
    else:
        logger.error("Stage 03: Post Processing failed")

    return success


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Stage 03: Post Processing")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to processed DataFrame Parquet from Stage 02 (optional; auto-resolved when omitted)",
    )
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config_yaml(args.config)

    if args.input:
        input_file = Path(args.input)
    else:
        input_dir = get_output_path("02_processed", "")
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return 1

        processed_name = config.get("pipeline_files", {}).get("stage02_processed_emg") or "processed_emg_data.parquet"
        sig_cfg = config.get("signal_processing", {}) or {}
        selected_option = sig_cfg.get("selected_option")
        try:
            input_file = _resolve_stage02_input_file(
                input_dir=input_dir,
                processed_name=processed_name,
                selected_option=selected_option,
            )
        except Exception as e:
            logger.error(f"Failed to resolve Stage02 input file: {e}")
            return 1

        if input_file is None:
            logger.error(f"No processed EMG data found. Expected '{processed_name}' in {input_dir}")
            return 1

    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_file}")
        return 1

    success = process_post_processing_steps(input_file, output_dir, config)
    if success:
        logger.info("Stage 03: Post Processing completed successfully")
        return 0

    logger.error("Stage 03: Post Processing failed")
    return 1


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SystemExit(0 if run() else 1)
    raise SystemExit(main())

