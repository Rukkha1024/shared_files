#!/usr/bin/env python3
"""
Stage 01 entrypoint.

Implementation lives in `src/stages/stage01_build_dataset.py`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is importable when executed directly.
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_helpers import get_output_path, load_config_yaml
from src.stages.stage01_build_dataset import DatasetBuilder, logger


def run():
    """Run function for pipeline integration."""
    output_dir = get_output_path("01_dataset", "")

    config = load_config_yaml()
    builder = DatasetBuilder(
        pre_frames=config["segmentation"]["pre_frames"],
        post_frames=config["segmentation"]["post_frames"],
    )

    return builder.run(output_dir)


def main() -> None:
    """Main entry point for standalone execution."""
    config = load_config_yaml()
    default_pre = config["segmentation"]["pre_frames"]
    default_post = config["segmentation"]["post_frames"]

    parser = argparse.ArgumentParser(
        description="Stage 01: Build Comprehensive Dataset (Unified Stage 01)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--pre-frames",
        type=int,
        default=default_pre,
        help=f"Pre-onset frames (default: {default_pre})",
    )
    parser.add_argument(
        "--post-frames",
        type=int,
        default=default_post,
        help=f"Post-offset frames (default: {default_post})",
    )

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    output_dir = get_output_path("01_dataset", "")

    builder = DatasetBuilder(
        pre_frames=args.pre_frames,
        post_frames=args.post_frames,
    )

    success = builder.run(output_dir)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()

