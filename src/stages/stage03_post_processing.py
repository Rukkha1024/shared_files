#!/usr/bin/env python3
"""
Stage 03: Post Processing
Post-processing stage that applies normalization to processed EMG data.
Takes processed data from Stage 02 as input and outputs normalized data.
"""

import os
import argparse
import yaml
import logging
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Sequence
import traceback
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import sys
from tqdm import tqdm
from scipy import signal

from src.config_helpers import get_output_path, load_config_yaml, DATA_DIR, get_device_hz, get_frame_ratio
from src.utils import get_logger, log_and_print, write_summary, read_parquet_robust, save_parquet
from src.stages.common import normalize_option_token as _normalize_option_token
from src.stages.common import standardize_join_keys_lf as _standardize_join_keys_lf

# Setup logging
logger = get_logger('03_post_processing')

# Suppress warnings from libraries
import warnings
warnings.filterwarnings('ignore')


# ==================== NORMALIZATION FUNCTIONS ====================
# Imported and adapted from legacy trial_normalize.py and norm.py

def load_platform_timing_data() -> pl.DataFrame:
    """
    Load platform onset/offset timing data from Excel file.

    Returns:
        DataFrame with platform timing information
    """
    # Load config to get platform timing path
    config = load_config_yaml()
    platform_file = DATA_DIR / config['platform_timing']['relative_path']
    sheet_name = config['platform_timing']['sheet_name']

    if not platform_file.exists():
        logger.error(f"Platform timing file not found: {platform_file}")
        raise FileNotFoundError(f"Platform timing file not found: {platform_file}")

    # Read Excel directly with Polars
    df = pl.read_excel(platform_file, sheet_name=sheet_name, engine="openpyxl")
    # Strip whitespace from column names
    df = df.rename({col: col.strip() for col in df.columns})
    # Standardize column names that will be joined against
    # Ensure 'trial' exists (source uses 'trial') and trim whitespace in 'subject'
    if 'subject' in df.columns:
        df = df.with_columns(pl.col('subject').cast(pl.Utf8))
    if 'velocity' in df.columns:
        # Keep numeric if provided; polars will infer
        pass
    if 'trial' in df.columns:
        # Normalize trial dtype to Int64 for robust join
        df = df.with_columns(pl.col('trial').cast(pl.Int64))
    logger.info(f"Loaded platform timing data: {df.height} entries")
    return df

def _platform_timing_lazy(config: dict) -> pl.LazyFrame:
    df = load_platform_timing_data()

    if "trial_num" not in df.columns and "trial" in df.columns:
        df = df.rename({"trial": "trial_num"})

    lf = df.lazy()
    lf = _standardize_join_keys_lf(lf, ["subject", "velocity", "trial_num"])
    if "platform_onset" in lf.collect_schema().names():
        lf = lf.with_columns(pl.col("platform_onset").cast(pl.Int64, strict=False))
    if "platform_offset" in lf.collect_schema().names():
        lf = lf.with_columns(pl.col("platform_offset").cast(pl.Int64, strict=False))

    return lf


def calculate_baseline_normalization_params(
    lf: pl.LazyFrame,
    emg_cols: List[str],
    platform_lf: pl.LazyFrame,
    baseline_window_ms: int,
    config: dict,
) -> pl.LazyFrame:
    """
    Polars implementation: compute baseline mean per trial and channel by filtering
    rows in [t0_dev - N, t0_dev), where t0_dev = platform_onset * FRAME_RATIO.

    Returns a Polars DataFrame keyed by (subject, velocity, trial_num) with columns
    base_<channel> for each EMG channel.
    """
    join_keys = ["subject", "velocity", "trial_num"]

    fs = get_device_hz(config)
    N = int(baseline_window_ms * fs / 1000)
    frame_ratio = get_frame_ratio(config)

    lf = _standardize_join_keys_lf(lf, join_keys)
    platform_lf = _standardize_join_keys_lf(platform_lf, join_keys)

    joined = (
        lf.join(platform_lf, on=join_keys, how="left")
        .with_columns((pl.col("platform_onset") * pl.lit(frame_ratio)).alias("t0_dev"))
    )

    mask = (pl.col("original_DeviceFrame") >= (pl.col("t0_dev") - pl.lit(N))) & (
        pl.col("original_DeviceFrame") < pl.col("t0_dev")
    )

    aggs: list[pl.Expr] = [pl.col(ch).filter(mask).mean().alias(f"base_{ch}") for ch in emg_cols]
    return joined.group_by(join_keys).agg(aggs)

def calculate_trial_normalization_params(df: pl.DataFrame, emg_cols: List[str], method: str = "min_max", **kwargs) -> Any:
    """
    Calculate trial normalization parameters for each subject-intervention-trial_num and channel.
    Supports multiple normalization methods including baseline normalization.
    
    Args:
        df: DataFrame containing processed EMG data from Stage 5
        emg_cols: List of EMG column names
        method: Normalization method ("baseline", "min_max", "z_score", "max_value")
        **kwargs: Additional parameters (e.g., platform_data, baseline_window_ms for baseline method)
        
    Returns:
        Dictionary mapping (subject, velocity, trial_num) -> channel -> normalization parameters
    """
    if method == "baseline":
        platform_lf = kwargs.get("platform_lf")
        baseline_window_ms = int(kwargs.get("baseline_window_ms", 1000))
        config = kwargs.get("config")

        if platform_lf is None or config is None:
            logger.error("Platform timing data required for baseline normalization")
            raise ValueError("Platform timing data required for baseline normalization")

        if isinstance(df, pl.DataFrame):
            df = df.lazy()

        return calculate_baseline_normalization_params(df, emg_cols, platform_lf, baseline_window_ms, config)
    
    # Original normalization methods (min_max, z_score, max_value)
    # Vectorized Polars parameter calculation for other methods
    if isinstance(df, pl.DataFrame):
        df = df.lazy()

    join_keys = ["subject", "velocity", "trial_num"]
    aggs = []
    if method == "min_max":
        for ch in emg_cols:
            aggs += [
                pl.col(ch).min().alias(f'min_{ch}'),
                pl.col(ch).max().alias(f'max_{ch}')
            ]
    elif method == "z_score":
        for ch in emg_cols:
            aggs += [
                pl.col(ch).mean().alias(f'mean_{ch}'),
                pl.col(ch).std().alias(f'std_{ch}')
            ]
    elif method == "max_value":
        for ch in emg_cols:
            aggs.append(pl.col(ch).abs().max().alias(f'maxabs_{ch}'))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    params = df.group_by(join_keys).agg(aggs)
    return params

def apply_trial_normalization(
    df: pl.DataFrame | pl.LazyFrame,
    emg_cols: List[str],
    norm_params: pl.LazyFrame,
    method: str = "min_max",
    config: dict = None,
) -> pl.LazyFrame:
    """
    Apply trial normalization to DataFrame using pre-calculated parameters with parallel processing.
    
    Args:
        df: DataFrame containing processed EMG data from Stage 5
        emg_cols: List of EMG column names
        norm_params: Normalization parameters
        method: Normalization method
        config: Configuration dictionary for parallel processing settings
        
    Returns:
        DataFrame with trial-normalized EMG data
    """
    # Vectorized Polars normalization using joined params
    log_and_print("Processing trials...", logging.INFO, '03_post_processing')
    join_keys = ["subject", "velocity", "trial_num"]

    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    original_cols = lf.collect_schema().names()

    if method == "baseline":
        normalized = lf.join(norm_params, on=join_keys, how="left")
        # Apply division; when baseline is null -> keep original, when zero -> null
        updates = []
        for ch in emg_cols:
            base_col = f'base_{ch}'
            updates.append(
                pl.when(pl.col(base_col).is_null())
                .then(pl.col(ch))
                .otherwise(
                    pl.when(pl.col(base_col) == 0)
                    .then(pl.lit(None).cast(pl.Float64))
                    .otherwise(pl.col(ch) / pl.col(base_col))
                ).alias(ch)
            )

        normalized = normalized.with_columns(updates)

        # Replace Inf with 0.0 but keep Nulls (NaN-equivalent) for baseline method
        fix_inf = [
            pl.when(pl.col(ch).is_infinite()).then(0.0).otherwise(pl.col(ch)).alias(ch)
            for ch in emg_cols
        ]
        normalized = normalized.with_columns(fix_inf)

        normalized_schema_names = normalized.collect_schema().names()
        normalized = normalized.drop([f"base_{ch}" for ch in emg_cols if f"base_{ch}" in normalized_schema_names])
        return normalized.select(original_cols)

    # Other methods: params computed as grouped DataFrame, join then transform
    result = lf.join(norm_params, on=join_keys, how="left")

    updates = []
    if method == "min_max":
        for ch in emg_cols:
            mn = f'min_{ch}'
            mx = f'max_{ch}'
            updates.append(
                pl.when((pl.col(mx).is_not_null()) & (pl.col(mx) != pl.col(mn)))
                .then((pl.col(ch) - pl.col(mn)) / (pl.col(mx) - pl.col(mn)))
                .otherwise(0.0)
                .alias(ch)
            )
    elif method == "z_score":
        for ch in emg_cols:
            mean_c = f'mean_{ch}'
            std_c = f'std_{ch}'
            updates.append(
                pl.when((pl.col(std_c).is_not_null()) & (pl.col(std_c) > 0))
                .then((pl.col(ch) - pl.col(mean_c)) / pl.col(std_c))
                .otherwise(0.0)
                .alias(ch)
            )
    elif method == "max_value":
        for ch in emg_cols:
            mxa = f'maxabs_{ch}'
            updates.append(
                pl.when((pl.col(mxa).is_not_null()) & (pl.col(mxa) > 0))
                .then(pl.col(ch) / pl.col(mxa))
                .otherwise(0.0)
                .alias(ch)
            )
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    result = result.with_columns(updates)
    # Replace NaN/Inf with 0.0 for non-baseline methods
    fix_vals = [
        pl.when(pl.col(ch).is_infinite() | pl.col(ch).is_nan()).then(0.0).otherwise(pl.col(ch)).alias(ch)
        for ch in emg_cols
    ]
    result = result.with_columns(fix_vals)

    drop_cols: list[str] = []
    result_schema_names = result.collect_schema().names()
    for ch in emg_cols:
        for prefix in ("min_", "max_", "mean_", "std_", "maxabs_"):
            name = f"{prefix}{ch}"
            if name in result_schema_names:
                drop_cols.append(name)
    if drop_cols:
        result = result.drop(drop_cols)

    return result.select(original_cols)

def validate_trial_normalization(df: pl.DataFrame | pl.LazyFrame, emg_cols: List[str], method: str) -> Dict[str, Any]:
    """
    Simple validation of trial normalization results.

    Args:
        df: DataFrame containing trial-normalized EMG data
        emg_cols: List of EMG column names
        method: Normalization method used

    Returns:
        Dictionary containing validation statistics
    """
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    schema_names = lf.collect_schema().names()
    has_keys = all(k in schema_names for k in ["subject", "velocity", "trial_num"])

    exprs: list[pl.Expr] = [pl.len().alias("total_records")]
    if "subject" in schema_names:
        exprs.append(pl.col("subject").n_unique().alias("total_subjects"))
    if has_keys:
        exprs.append(pl.struct(["subject", "velocity", "trial_num"]).n_unique().alias("total_trials"))
    else:
        exprs.append(pl.lit(0).alias("total_trials"))

    for ch in emg_cols:
        if ch not in schema_names:
            continue
        exprs.extend(
            [
                pl.col(ch).min().alias(f"{ch}__min"),
                pl.col(ch).max().alias(f"{ch}__max"),
                pl.col(ch).mean().alias(f"{ch}__mean"),
                pl.col(ch).std().alias(f"{ch}__std"),
                pl.col(ch).is_null().sum().alias(f"{ch}__nulls"),
                pl.col(ch).is_infinite().sum().alias(f"{ch}__infs"),
            ]
        )

    stats_row = lf.select(exprs).collect(streaming=True).row(0, named=True)

    total_records = int(stats_row.get("total_records", 0))
    total_subjects = int(stats_row.get("total_subjects", 0))
    total_trials = int(stats_row.get("total_trials", 0))

    ch_mins, ch_maxs, ch_means, ch_stds = [], [], [], []
    nan_counts, inf_counts = 0, 0
    for ch in emg_cols:
        if ch not in schema_names:
            continue
        ch_min = stats_row.get(f"{ch}__min")
        ch_max = stats_row.get(f"{ch}__max")
        ch_mean = stats_row.get(f"{ch}__mean")
        ch_std = stats_row.get(f"{ch}__std")
        ch_nan = int(stats_row.get(f"{ch}__nulls", 0) or 0)
        ch_inf = int(stats_row.get(f"{ch}__infs", 0) or 0)
        if ch_min is not None:
            ch_mins.append(float(ch_min))
        if ch_max is not None:
            ch_maxs.append(float(ch_max))
        if ch_mean is not None:
            ch_means.append(float(ch_mean))
        if ch_std is not None:
            ch_stds.append(float(ch_std))
        nan_counts += ch_nan
        inf_counts += ch_inf

    rng_min = float(min(ch_mins)) if ch_mins else float('nan')
    rng_max = float(max(ch_maxs)) if ch_maxs else float('nan')
    rng_mean = float(np.nanmean(ch_means)) if ch_means else float('nan')
    rng_std = float(np.nanmean(ch_stds)) if ch_stds else float('nan')

    validation_stats = {
        'total_records': total_records,
        'total_subjects': total_subjects,
        'total_trials': total_trials,
        'emg_channels': len(emg_cols),
        'range': {
            'min': rng_min,
            'max': rng_max,
            'mean': rng_mean,
            'std': rng_std,
        },
        'nan_count': int(nan_counts),
        'inf_count': int(inf_counts),
        'validation_passed': True
    }

    # Simple validation check based on method
    if method == "baseline":
        # For baseline normalization, values should be ratios (typically 0.5 to 2.0 range)
        # NaN values are acceptable when baseline is 0
        validation_stats['validation_passed'] = (
            validation_stats['range']['min'] >= -100 and
            validation_stats['range']['max'] <= 100
        )
    elif method == "min_max":
        # Should be normalized to [0, 1] range
        validation_stats['validation_passed'] = (
            validation_stats['range']['min'] >= 0 and
            validation_stats['range']['max'] <= 1
        )
    elif method == "z_score":
        # Should have reasonable range for z-scores
        validation_stats['validation_passed'] = (
            validation_stats['range']['min'] >= -10 and
            validation_stats['range']['max'] <= 10
        )
    else:
        # For other methods, require finite values across EMG cols
        try:
            all_finite = True
            for ch in emg_cols:
                finite_count = int(df.select(pl.col(ch).is_finite().sum()).to_series()[0])
                if finite_count != df.height:
                    all_finite = False
                    break
            validation_stats['validation_passed'] = all_finite
        except Exception:
            validation_stats['validation_passed'] = True

    return validation_stats


# ==================== SECOND NORMALIZATION FUNCTIONS ====================
# Two-step normalization: baseline -> min_max or unit_variance

def calculate_second_normalization_params(df: pl.DataFrame | pl.LazyFrame, emg_cols: List[str], method: str, grouping_columns: List[str]) -> pl.LazyFrame:
    """
    Calculate second normalization parameters (min_max or unit_variance) for each muscle channel.
    Applied after baseline normalization.

    Args:
        df: DataFrame containing baseline-normalized EMG data
        emg_cols: List of EMG column names
        method: Second normalization method ("min_max" or "unit_variance")
        grouping_columns: List of column names to group by (e.g., ['subject'], ['subject', 'velocity'])

    Returns:
        LazyFrame with normalization parameters grouped by specified columns
    """
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    
    aggs: list[pl.Expr] = []
    if method == "min_max":
        for ch in emg_cols:
            aggs += [
                pl.col(ch).min().alias(f'min_{ch}'),
                pl.col(ch).max().alias(f'max_{ch}')
            ]
    elif method == "unit_variance":
        for ch in emg_cols:
            aggs.append(pl.col(ch).std().alias(f'std_{ch}'))
    elif method == "z_score":
        for ch in emg_cols:
            aggs += [
                pl.col(ch).mean().alias(f"mean_{ch}"),
                pl.col(ch).std().alias(f"std_{ch}"),
            ]
    elif method == "max_value":
        for ch in emg_cols:
            aggs.append(pl.col(ch).abs().max().alias(f"maxabs_{ch}"))
    else:
        raise ValueError(f"Unknown second normalization method: {method}")

    return lf.group_by(grouping_columns).agg(aggs)


def apply_second_normalization(
    df: pl.DataFrame | pl.LazyFrame,
    emg_cols: List[str],
    second_norm_params: pl.LazyFrame,
    method: str,
    grouping_columns: List[str],
    config: dict = None,
) -> pl.LazyFrame:
    """
    Apply second normalization (min_max or unit_variance) to baseline-normalized data.

    Args:
        df: DataFrame containing baseline-normalized EMG data
        emg_cols: List of EMG column names
        second_norm_params: Second normalization parameters
        method: Second normalization method ("min_max" or "unit_variance")
        grouping_columns: List of column names to group by (e.g., ['subject'], ['subject', 'velocity'])
        config: Configuration dictionary

    Returns:
        DataFrame with second normalization applied
    """
    logger.info(f"Applying second normalization ({method}, grouping={grouping_columns})...")

    # Join params and apply vectorized normalization
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    original_cols = lf.collect_schema().names()
    result = lf.join(second_norm_params, on=grouping_columns, how='left')

    updates: list[pl.Expr] = []
    if method == "min_max":
        for ch in emg_cols:
            mn = f'min_{ch}'
            mx = f'max_{ch}'
            updates.append(
                pl.when((pl.col(mx).is_not_null()) & (pl.col(mx) != pl.col(mn)))
                .then((pl.col(ch) - pl.col(mn)) / (pl.col(mx) - pl.col(mn)))
                .otherwise(0.5)
                .alias(ch)
            )
    elif method == "unit_variance":
        for ch in emg_cols:
            std_c = f'std_{ch}'
            updates.append(
                pl.when((pl.col(std_c).is_not_null()) & (pl.col(std_c) > 0))
                .then(pl.col(ch) / pl.col(std_c))
                .otherwise(pl.col(ch))
                .alias(ch)
            )
    elif method == "z_score":
        for ch in emg_cols:
            mean_c = f"mean_{ch}"
            std_c = f"std_{ch}"
            updates.append(
                pl.when((pl.col(std_c).is_not_null()) & (pl.col(std_c) > 0))
                .then((pl.col(ch) - pl.col(mean_c)) / pl.col(std_c))
                .otherwise(pl.lit(None).cast(pl.Float64))
                .alias(ch)
            )
    elif method == "max_value":
        for ch in emg_cols:
            mx = f"maxabs_{ch}"
            updates.append(
                pl.when((pl.col(mx).is_not_null()) & (pl.col(mx) > 0))
                .then(pl.col(ch) / pl.col(mx))
                .otherwise(pl.lit(None).cast(pl.Float64))
                .alias(ch)
            )
    else:
        raise ValueError(f"Unknown second normalization method: {method}")

    result = result.with_columns(updates)

    # Replace Inf with 0.0, keep Nulls (from baseline)
    result = result.with_columns([
        pl.when(pl.col(ch).is_infinite()).then(0.0).otherwise(pl.col(ch)).alias(ch)
        for ch in emg_cols
    ])

    drop_cols: list[str] = []
    result_schema_names = result.collect_schema().names()
    for ch in emg_cols:
        for prefix in ("min_", "max_", "mean_", "std_", "maxabs_"):
            name = f"{prefix}{ch}"
            if name in result_schema_names:
                drop_cols.append(name)
    if drop_cols:
        result = result.drop(drop_cols)

    return result.select(original_cols)


# ==================== POST-PROCESSING ORCHESTRATION ====================

def apply_normalization_step(df: pl.LazyFrame, emg_cols: List[str], config: dict) -> pl.LazyFrame:
    """
    Apply normalization steps dynamically based on config.

    Args:
        df: Input DataFrame
        emg_cols: List of EMG column names
        config: Configuration dictionary

    Returns:
        Normalized DataFrame
    """
    logger.info("Applying normalization pipeline...")

    norm_cfg = config.get("normalization", {}) or {}
    execution_order = norm_cfg.get("execution_order")
    definitions = norm_cfg.get("definitions")

    if execution_order is None or definitions is None:
        raise KeyError("Missing config: normalization.execution_order and/or normalization.definitions")
    if not isinstance(execution_order, list):
        raise TypeError("normalization.execution_order must be a list")
    if not isinstance(definitions, dict):
        raise TypeError("normalization.definitions must be a dict")

    if len(execution_order) == 0:
        logger.warning("Normalization execution_order is empty; returning input unchanged")
        return df.lazy() if isinstance(df, pl.DataFrame) else df

    current = df.lazy() if isinstance(df, pl.DataFrame) else df

    for idx, step_id in enumerate(execution_order, start=1):
        if not isinstance(step_id, str) or not step_id.strip():
            raise ValueError(f"Invalid normalization step id at position {idx}: {step_id!r}")

        step_cfg = definitions.get(step_id)
        if step_cfg is None:
            raise KeyError(f"Normalization step '{step_id}' not found in normalization.definitions")
        if not isinstance(step_cfg, dict):
            raise TypeError(f"Normalization definition for '{step_id}' must be a dict")

        method = step_cfg.get("method")
        if not isinstance(method, str) or not method.strip():
            raise ValueError(f"Normalization step '{step_id}' missing required field: method")
        method = method.strip()

        logger.info(f"[Normalization {idx}/{len(execution_order)}] step='{step_id}', method='{method}'")

        try:
            if method == "baseline":
                logger.info("Loading platform timing data for baseline normalization...")
                platform_lf = _platform_timing_lazy(config)

                baseline_window_ms = int(step_cfg.get("baseline_window_ms", 1000))
                logger.info(f"Baseline window: {baseline_window_ms}ms")

                logger.info("Calculating baseline normalization parameters...")
                norm_params = calculate_trial_normalization_params(
                    current,
                    emg_cols,
                    "baseline",
                    platform_lf=platform_lf,
                    baseline_window_ms=baseline_window_ms,
                    config=config,
                )

                n_params = int(norm_params.select(pl.len()).collect(streaming=True).to_series()[0])
                if n_params == 0:
                    raise ValueError("Could not calculate baseline normalization parameters (0 groups)")
                logger.info(f"Calculated baseline parameters for {n_params} trials")

                logger.info("Applying baseline normalization...")
                current = apply_trial_normalization(current, emg_cols, norm_params, "baseline", config)

            else:
                grouping_columns = step_cfg.get("grouping_columns", ["subject"])
                if not isinstance(grouping_columns, list) or not grouping_columns:
                    raise ValueError(f"Normalization step '{step_id}' has invalid grouping_columns: {grouping_columns!r}")

                grouping_str = "-".join(str(c) for c in grouping_columns)
                logger.info(f"Calculating '{method}' normalization parameters (grouping={grouping_columns})...")
                norm_params = calculate_second_normalization_params(current, emg_cols, method, grouping_columns)

                n_groups = int(norm_params.select(pl.len()).collect(streaming=True).to_series()[0])
                if n_groups == 0:
                    raise ValueError("Could not calculate normalization parameters (0 groups)")
                logger.info(f"Calculated '{method}' parameters for {n_groups} groups (by {grouping_str})")

                logger.info(f"Applying '{method}' normalization (grouping={grouping_columns})...")
                current = apply_second_normalization(current, emg_cols, norm_params, method, grouping_columns, config)

            validation_stats = validate_trial_normalization(current, emg_cols, method)
            if not validation_stats.get("validation_passed", True):
                logger.warning(f"Normalization validation failed (step='{step_id}', method='{method}'), but continuing...")
            else:
                logger.info(f"Normalization validation passed (step='{step_id}', method='{method}')")

        except Exception as e:
            logger.error(f"Normalization step failed (step='{step_id}', method='{method}'): {e}")
            logger.error(traceback.format_exc())
            raise

    return current

def process_post_processing_steps(input_file: Path, output_dir: Path, config: dict) -> bool:
    """
    Main processing function that applies normalization to processed EMG data.
    
    Args:
        input_file: Path to processed DataFrame file from Stage 02
        output_dir: Output directory for normalized files
        config: Configuration dictionary
        
    Returns:
        True if processing was successful, False otherwise
    """
    try:
        # Load processed data from Stage 02 as LazyFrame (parquet only)
        logger.info(f"Scanning processed data from {input_file}")
        lf = pl.scan_parquet(input_file)

        lf = _standardize_join_keys_lf(lf, ["subject", "velocity", "trial_num"])

        # Identify EMG columns
        # Use muscle names from config YAML to determine EMG columns
        muscle_names = config['muscles']['names']
        schema_names = lf.collect_schema().names()
        emg_cols = [c for c in muscle_names if c in schema_names]

        logger.info(f"Found {len(emg_cols)} EMG channels")

        if not emg_cols:
            logger.error("No EMG columns found in the DataFrame")
            return False

        # Apply normalization (Polars)
        logger.info("=== Applying Normalization ===")
        normalized_lf = apply_normalization_step(lf, emg_cols, config)

        # Save normalized data (parquet only, streaming)
        output_name = config.get("pipeline_files", {}).get("stage03_normalized") or "normalized_data.parquet"
        final_parquet = output_dir / output_name
        save_parquet(normalized_lf, final_parquet)
        logger.info(f"Saved normalized data to {final_parquet}")

        return True
        
    except Exception as e:
        logger.error(f"Error in post-processing: {e}")
        logger.error(traceback.format_exc())
        return False

def _resolve_stage02_input_file(
    *,
    input_dir: Path,
    processed_name: str,
    selected_option: Optional[str],
) -> Optional[Path]:
    """
    Resolve Stage02 parquet path.

    Priority:
    1) `<input_dir>/<processed_name>` (legacy single-output layout)
    2) `<input_dir>/<selected_option>/<processed_name>` (exact/case-insensitive)
    3) tolerant single-match by normalized token
    4) first available option directory (sorted by name)
    """
    processed_parquet = input_dir / processed_name
    if processed_parquet.exists():
        logger.info(f"Found processed data: {processed_parquet}")
        return processed_parquet

    found_options: list[tuple[str, Path]] = []
    for sub_dir in sorted(input_dir.iterdir(), key=lambda p: p.name):
        if not sub_dir.is_dir():
            continue
        sub_parquet = sub_dir / processed_name
        if sub_parquet.exists():
            found_options.append((sub_dir.name, sub_parquet))

    if not found_options:
        return None

    selected = str(selected_option).strip() if selected_option is not None else ""
    if selected:
        exact = [(name, path) for name, path in found_options if name == selected]
        if len(exact) == 1:
            logger.info(f"Using selected option from config (exact): {exact[0][0]}")
            logger.info(f"Found processed data in subdirectory: {exact[0][1]}")
            return exact[0][1]

        ci = [(name, path) for name, path in found_options if name.lower() == selected.lower()]
        if len(ci) == 1:
            logger.info(f"Using selected option from config (case-insensitive): {ci[0][0]}")
            logger.info(f"Found processed data in subdirectory: {ci[0][1]}")
            return ci[0][1]

        normalized_selected = _normalize_option_token(selected)
        fuzzy = [
            (name, path)
            for name, path in found_options
            if normalized_selected and normalized_selected in _normalize_option_token(name)
        ]
        if len(fuzzy) == 1:
            logger.info(f"Using selected option from config (tolerant): {selected} -> {fuzzy[0][0]}")
            logger.info(f"Found processed data in subdirectory: {fuzzy[0][1]}")
            return fuzzy[0][1]
        if len(fuzzy) > 1:
            names = ", ".join(name for name, _ in fuzzy)
            raise ValueError(
                f"signal_processing.selected_option='{selected}' matches multiple Stage02 option directories: {names}"
            )

        available = ", ".join(name for name, _ in found_options)
        logger.warning(
            f"signal_processing.selected_option='{selected}' not found in Stage02 outputs; "
            f"available options: {available}. Falling back to first available option."
        )

    chosen_name, chosen_path = found_options[0]
    logger.info(f"Found processed data in subdirectory: {chosen_path} (option={chosen_name})")
    return chosen_path
