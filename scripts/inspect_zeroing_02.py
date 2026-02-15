#!/usr/bin/env python3
"""
Diagnostic script for Stage 02 zero-point adjustment.

Loads the merged dataset and reports, for each group of
columns (forceplate, COP, COM), how close the per-trial baseline means
are to zero.

This helps verify that zero-point adjustment is applied as expected.
"""

from pathlib import Path
from typing import List, Dict, Any
import sys

import polars as pl

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_helpers import get_output_path, load_config_yaml


def compute_baseline_stats(
    df: pl.DataFrame,
    group_name: str,
    columns: List[str],
    window_rows: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute per-column baseline statistics over the first `window_rows`
    samples of each trial.
    """
    available_cols = [c for c in columns if c in df.columns]
    if not available_cols:
        print(f"[INFO] No columns from group '{group_name}' found in dataset.")
        return {}

    join_keys = ["subject", "velocity", "trial_num"]

    aggs = []
    for col in available_cols:
        aggs.append(
            pl.col(col)
            .cast(pl.Float64)
            .head(window_rows)
            .mean()
            .alias(f"{col}_baseline")
        )

    per_trial = df.group_by(join_keys).agg(aggs)

    stats: Dict[str, Dict[str, Any]] = {}
    for col in available_cols:
        base_col = f"{col}_baseline"
        series = per_trial[base_col]
        col_min = series.min()
        col_max = series.max()
        col_mean = series.mean()
        stats[col] = {
            "min": float(col_min) if col_min is not None else float("nan"),
            "max": float(col_max) if col_max is not None else float("nan"),
            "mean": float(col_mean) if col_mean is not None else float("nan"),
        }

    return stats


def print_stats(group_name: str, stats: Dict[str, Dict[str, Any]]) -> None:
    """Pretty-print baseline statistics for a group."""
    if not stats:
        return

    print(f"\n=== Baseline statistics for {group_name} columns ===")
    for col, values in stats.items():
        min_val = values["min"]
        max_val = values["max"]
        mean_val = values["mean"]
        print(
            f"- {col}: "
            f"min={min_val:.6f}, mean={mean_val:.6f}, max={max_val:.6f}"
        )


def main() -> int:
    config = load_config_yaml("config.yaml")
    dataset_path = get_output_path("01_dataset", config["pipeline_files"]["stage01_merged_dataset"])

    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        print("        Run Stage 01 before using this diagnostic script.")
        return 1

    print(f"[INFO] Loading dataset from: {dataset_path}")
    df = pl.read_parquet(dataset_path)

    forceplate_cfg = config.get("forceplate", {}).get("zeroing", {})
    com_cfg = config.get("com", {}).get("zeroing", {})

    fp_cols = forceplate_cfg.get("columns", [])
    fp_rows = int(forceplate_cfg.get("rows", 0))
    com_cols = com_cfg.get("columns", [])
    com_rows = int(com_cfg.get("rows", 0))

    # Compute and print baseline stats for each group
    fp_stats = compute_baseline_stats(
        df,
        "forceplate",
        fp_cols,
        fp_rows,
    )
    print_stats("forceplate", fp_stats)

    com_stats = compute_baseline_stats(
        df,
        "COM",
        com_cols,
        com_rows,
    )
    print_stats("COM", com_stats)

    print("\n[INFO] Zero-point diagnostic completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
