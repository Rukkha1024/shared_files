#!/usr/bin/env python3
"""
Stage 02 entrypoint.

Implementation lives in `src/stages/stage02_emg_filtering.py`.
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

# Ensure the project root is importable when executed directly.
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_helpers import get_output_path, load_config_yaml
from src.stages.stage02_emg_filtering import StageRunner


def run():
    """Run function for pipeline integration."""
    config = load_config_yaml("config.yaml")
    parquet_name = config.get("pipeline_files", {}).get("stage01_merged_dataset") or "merged_data_comprehensive.parquet"
    runner = StageRunner(
        input_path=get_output_path("01_dataset", parquet_name),
        output_dir=get_output_path("02_processed", ""),
        config_path="config.yaml",
    )
    return runner.execute()


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 02: EMG Signal Filtering")
    cfg = load_config_yaml("config.yaml")
    parquet_name = cfg.get("pipeline_files", {}).get("stage01_merged_dataset") or "merged_data_comprehensive.parquet"
    parser.add_argument(
        "--input",
        type=str,
        default=str(get_output_path("01_dataset", parquet_name)),
        help="Input parquet file from Stage 01",
    )
    parser.add_argument("--output", type=str, default=str(get_output_path("02_processed", "")), help="Output directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    runner = StageRunner(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        config_path=args.config,
        debug=args.debug,
    )
    success = runner.execute()
    return 0 if success else 1


if __name__ == "__main__":
    if sys.platform == "win32":
        multiprocessing.set_start_method("spawn", force=True)
    raise SystemExit(main())

