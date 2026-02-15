#!/usr/bin/env python3
"""
Stage 01: Build Comprehensive Dataset (Unified Stage 01)
================================================================

This unified stage replaces the previous pipeline:
- Stage 01: Platform Segmentation
- Stage 02: Build EMG Dataset
- Stage 03: Forceplate Processing

New unified workflow:
1. Load perturbation parquet file
2. Load platform timing and metadata
3. Segment trials based on platform timing (EMG + Forceplate together)
4. Convert EMG column names (Dev1/ai0 → TA, etc.)
5. Merge platform metadata (state, T/F_step, etc.)
6. Save comprehensive Parquet directly

Benefits:
- Single I/O pass (load parquet once, save Parquet once)
- No intermediate files (PLATFORM_SEGMENTS eliminated)
- Simpler pipeline (3 stages → 1 stage)
- Easier maintenance and debugging

Author: Generated with Claude Code
Date: 2025-01-18
"""

import os
import sys
import logging
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import traceback
import re
import unicodedata
from datetime import datetime
from dataclasses import dataclass

from src.config_helpers import (
    MODULE_DIR, load_config_yaml, DATA_DIR, get_output_path, get_frame_ratio
)
from src.utils import get_logger, create_output_directory, save_parquet

def _nearest_index(arr: np.ndarray, value: int) -> int:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return 0
    idx = int(np.nanargmin(np.abs(arr - float(value))))
    return max(0, min(idx, int(arr.size) - 1))


@dataclass(frozen=True)
class _SubtractTemplate100Hz:
    velocity_int: int
    unload_range_frames: int
    fx: np.ndarray
    fy: np.ndarray
    fz: np.ndarray
    mx: np.ndarray
    my: np.ndarray
    mz: np.ndarray
    n_trials: int
    meta: dict[str, Any]



# Setup logging
logger = get_logger('01_build_dataset')

# Ensure Polars uses all available threads unless user overrides
os.environ.setdefault("POLARS_MAX_THREADS", str(os.cpu_count() or 16))


class DatasetBuilder:
    """Unified dataset builder that combines functionality."""

    def __init__(self,
                 pre_frames: int = None,
                 post_frames: int = None,
                 config_path: str | None = None):
        """
        Initialize the DatasetBuilder.

        Args:
            pre_frames: Number of frames before platform onset (from config if None)
            post_frames: Number of frames after platform offset (from config if None)
            config_path: Optional path to a YAML config file (default: MODULE_DIR/config.yaml)
        """
        # Load configuration first
        self.config = load_config_yaml(config_path)

        # Use config values if not provided
        if pre_frames is None:
            pre_frames = self.config['segmentation']['pre_frames']
        if post_frames is None:
            post_frames = self.config['segmentation']['post_frames']

        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.muscle_names = self._load_muscle_names()

        # Load data sources
        self.parquet_lf = self._scan_parquet_data()
        self.platform_timing_df = self._load_platform_timing()
        self.platform_metadata_df = self._load_platform_metadata()

        # Statistics
        self.stats = {
            'total_trials': 0,
            'segments_created': 0,
            'errors': []
        }

        # Lazy-built cache: velocity_int -> unloaded mean template (100Hz)
        self._subtract_templates_cache: dict[int, _SubtractTemplate100Hz] | None = None

        logger.info("DatasetBuilder initialized successfully")

    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode text to NFC form for cross-platform compatibility."""
        if not text:
            return text
        return unicodedata.normalize('NFC', str(text).strip())

    def _load_muscle_names(self) -> List[str]:
        """Load muscle names from config.yaml."""
        try:
            muscle_names = self.config['muscles']['names']
            logger.info(f"Loaded {len(muscle_names)} muscle names from config")
            return muscle_names
        except KeyError as e:
            logger.error(f"Could not find muscle names in config: {e}")
            raise

    def _get_frame_schema_names(self, frame: pl.LazyFrame | pl.DataFrame) -> list[str]:
        if isinstance(frame, pl.LazyFrame):
            return frame.collect_schema().names()
        return list(frame.columns)

    def _forceplate_axis_transform_exprs(self, frame_cols: list[str]) -> list[pl.Expr]:
        fp_cfg = self.config.get("forceplate", {}) or {}
        axis_cfg = fp_cfg.get("axis_transform", {}) or {}
        if not bool(axis_cfg.get("enabled", False)):
            return []

        mapping = axis_cfg.get("mapping", {}) or {}
        if not isinstance(mapping, dict):
            raise ValueError("forceplate.axis_transform.mapping는 dict 이어야 합니다.")

        exprs: list[pl.Expr] = []
        for target, spec in mapping.items():
            target_name = str(target)
            if target_name not in frame_cols:
                continue
            if isinstance(spec, dict):
                source = str(spec.get("source", target_name))
                scale = spec.get("scale", 1.0)
            else:
                source = str(spec)
                scale = 1.0
            if source not in frame_cols:
                continue
            try:
                scale_val = float(scale)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"forceplate.axis_transform.mapping[{target_name}].scale must be numeric"
                ) from exc
            exprs.append(
                (pl.col(source).cast(pl.Float64, strict=False) * pl.lit(scale_val)).alias(target_name)
            )
        return exprs

    def _apply_forceplate_axis_transform(
        self,
        frame: pl.LazyFrame | pl.DataFrame,
    ) -> pl.LazyFrame | pl.DataFrame:
        frame_cols = self._get_frame_schema_names(frame)
        exprs = self._forceplate_axis_transform_exprs(frame_cols)
        if not exprs:
            return frame
        return frame.with_columns(exprs)

    def _scan_parquet_data(self) -> pl.LazyFrame:
        """Scan perturbation parquet file lazily and filter for perturb task."""
        try:
            parquet_cfg = self.config.get("perturb_parquet", {})
            path_str = parquet_cfg.get("path")

            if not path_str:
                raise ValueError("Stage 01 requires perturb_parquet.path in config.yaml")

            raw_path = Path(path_str)
            parquet_path = raw_path if raw_path.is_absolute() else (MODULE_DIR / raw_path).resolve()

            if not parquet_path.exists():
                raise FileNotFoundError(f"Perturbation parquet not found: {parquet_path}")

            logger.info(f"Scanning perturbation parquet from {parquet_path}")
            lf = pl.scan_parquet(parquet_path)

            # Filter for perturb task only if task column exists
            lf_cols = lf.collect_schema().names()
            if "task" in lf_cols:
                lf = lf.filter(pl.col("task") == "perturb")

            # Normalize key dtypes for robust joins
            lf = lf.with_columns([
                pl.col("subject").cast(pl.Utf8),
                pl.col("velocity").cast(pl.Float64, strict=False),
                pl.col("trial_num").cast(pl.Int64, strict=False),
                pl.col("MocapFrame").cast(pl.Int64, strict=False),
                pl.col("DeviceFrame").cast(pl.Int64, strict=False),
            ])

            lf = self._apply_forceplate_axis_transform(lf)

            return lf

        except Exception as e:
            logger.error(f"Failed to load parquet data: {e}")
            raise

    def _load_platform_timing(self) -> pl.DataFrame:
        """Load platform timing data (onset/offset) from Excel file."""
        try:
            # Build platform timing path from config
            platform_timing_path = DATA_DIR / self.config['platform_timing']['relative_path']
            sheet_name = self.config['platform_timing']['sheet_name']

            if not platform_timing_path.exists():
                raise FileNotFoundError(f"Platform timing file not found: {platform_timing_path}")

            # Read Excel directly with Polars
            df = pl.read_excel(platform_timing_path, sheet_name=sheet_name, engine="openpyxl")

            # Strip whitespace from column names
            df = df.rename({col: col.strip() for col in df.columns})

            # Normalize subject names
            df = df.with_columns(
                pl.col('subject').cast(pl.Utf8).map_elements(
                    lambda x: self.normalize_unicode(str(x)),
                    return_dtype=pl.Utf8
                )
            )

            logger.info(f"Loaded platform timing: {df.height} entries")
            return df

        except Exception as e:
            logger.error(f"Error loading platform timing: {e}")
            raise

    def _load_platform_metadata(self) -> pl.DataFrame:
        """Load platform metadata with all columns (state, T/F_step, etc.)."""
        try:
            # Build platform timing path from config
            platform_timing_path = DATA_DIR / self.config['platform_timing']['relative_path']
            sheet_name = self.config['platform_timing']['sheet_name']

            # Read Excel directly with Polars
            platform_df = pl.read_excel(platform_timing_path, sheet_name=sheet_name, engine="openpyxl")

            # Strip whitespace from column names
            platform_df = platform_df.rename({col: col.strip() for col in platform_df.columns})

            # Rename step_T/F to T/F_step for consistency
            if 'step_T/F' in platform_df.columns:
                platform_df = platform_df.rename({'step_T/F': 'T/F_step'})

            # Normalize subject names
            if 'subject' in platform_df.columns:
                platform_df = platform_df.with_columns(
                    pl.col('subject').cast(pl.Utf8).map_elements(
                        lambda x: self.normalize_unicode(str(x)),
                        return_dtype=pl.Utf8
                    )
                )

            # Ensure numeric types for merge keys
            if 'velocity' in platform_df.columns:
                platform_df = platform_df.with_columns(
                    pl.col('velocity').cast(pl.Float64, strict=False)
                )
            if 'trial' in platform_df.columns:
                platform_df = platform_df.with_columns(
                    pl.col('trial').cast(pl.Int64, strict=False)
                )

            logger.info(f"Loaded platform metadata: {platform_df.height} records (all columns preserved)")
            return platform_df

        except Exception as e:
            logger.error(f"Failed to load platform metadata: {e}")
            raise

    def build_dataset_lazy(self) -> pl.LazyFrame:
        """
        Build the complete dataset as a LazyFrame (streamable).

        Returns:
            LazyFrame of the comprehensive dataset (not collected).
        """
        logger.info("Building comprehensive dataset (lazy, streamable)...")

        keys = ["subject", "velocity", "trial_num"]
        frame_ratio = get_frame_ratio(self.config)

        plat = self.platform_timing_df.clone().with_row_index("plat_order")
        if "trial_num" not in plat.columns and "trial" in plat.columns:
            plat = plat.rename({"trial": "trial_num"})

        plat = plat.with_columns([
            pl.col("subject").cast(pl.Utf8),
            pl.col("velocity").cast(pl.Float64, strict=False),
            pl.col("trial_num").cast(pl.Int64, strict=False),
            pl.col("platform_onset").cast(pl.Int64, strict=False),
            pl.col("platform_offset").cast(pl.Int64, strict=False),
        ]).with_columns([
            (pl.col("platform_onset") * frame_ratio).alias("onset_device"),
            ((pl.col("platform_offset") + 1) * frame_ratio - 1).alias("offset_device"),
        ]).with_columns([
            (pl.col("onset_device") - int(self.pre_frames)).alias("start_frame_req"),
            (pl.col("offset_device") + int(self.post_frames)).alias("end_frame_req"),
        ])

        raw_cols = self.parquet_lf.collect_schema().names()
        emg_pattern = re.compile(r"Dev1/ai(\d+)")
        rename_map: Dict[str, str] = {}
        for col in raw_cols:
            m = emg_pattern.match(col)
            if m:
                idx = int(m.group(1))
                if idx < len(self.muscle_names):
                    rename_map[col] = self.muscle_names[idx]
                else:
                    logger.warning(f"Channel index {idx} exceeds muscle names list length")

        # Legacy definition: original_DeviceFrame is the raw absolute DeviceFrame (1000Hz)
        lf = self.parquet_lf.with_columns(
            pl.col("DeviceFrame").cast(pl.Int64, strict=False).alias("original_DeviceFrame")
        )

        joined = lf.join(
            plat.lazy(),
            on=keys,
            how="inner",
        )

        bounds_lf = joined.group_by(keys).agg([
            pl.first("start_frame_req").alias("start_frame_req"),
            pl.first("end_frame_req").alias("end_frame_req"),
            pl.first("plat_order").alias("plat_order"),
            pl.col("original_DeviceFrame").min().alias("min_raw"),
            pl.col("original_DeviceFrame").max().alias("max_raw"),
        ]).with_columns([
            pl.when(pl.col("start_frame_req") < pl.col("min_raw"))
            .then(pl.col("min_raw"))
            .otherwise(pl.col("start_frame_req"))
            .alias("start_frame"),
            pl.when(pl.col("end_frame_req") > pl.col("max_raw"))
            .then(pl.col("max_raw"))
            .otherwise(pl.col("end_frame_req"))
            .alias("end_frame"),
            (
                (pl.col("max_raw") >= pl.col("start_frame_req"))
                & (pl.col("min_raw") <= pl.col("end_frame_req"))
            ).alias("has_overlap"),
        ])

        segmented = joined.join(
            bounds_lf.select(keys + ["start_frame", "end_frame", "has_overlap"]),
            on=keys,
            how="left",
        ).filter(
            pl.col("has_overlap") == True
        ).filter(
            (pl.col("original_DeviceFrame") >= pl.col("start_frame"))
            & (pl.col("original_DeviceFrame") <= pl.col("end_frame"))
        )

        com_cfg = self.config.get("com", {}).get("zeroing", {})
        com_cols_cfg = com_cfg.get("columns", [])
        com_rows_cfg = int(com_cfg.get("rows", 3))

        zero_cols = [c for c in com_cols_cfg if c in raw_cols]
        baseline_exprs = []
        for c in zero_cols:
            baseline_exprs.append(
                pl.col(c)
                .cast(pl.Float64, strict=False)
                .filter(pl.col("original_DeviceFrame") < (pl.col("start_frame") + pl.lit(int(com_rows_cfg))))
                .mean()
                .alias(f"baseline_{c}")
            )

        if baseline_exprs:
            baselines_lf = segmented.group_by(keys).agg(baseline_exprs)
            segmented = segmented.join(baselines_lf, on=keys, how="left")
            updates = []
            for c in zero_cols:
                base_c = f"baseline_{c}"
                updates.append(
                    pl.when(pl.col(base_c).is_not_null())
                    .then(pl.col(c).cast(pl.Float64, strict=False) - pl.col(base_c))
                    .otherwise(pl.col(c).cast(pl.Float64, strict=False))
                    .alias(c)
                )
            segmented = segmented.with_columns(updates).drop([f"baseline_{c}" for c in zero_cols])

        segmented = segmented.with_columns([
            pl.lit("perturb").alias("intervention"),
            (pl.col("original_DeviceFrame") - pl.col("start_frame"))
            .cast(pl.Int64)
            .alias("DeviceFrame"),
        ])

        if rename_map:
            segmented = segmented.rename(rename_map)

        self.stats["total_trials"] = int(plat.height)
        created_stats = self._validate_output_lazy(segmented)

        keys_df = plat.select(keys).unique().sort(keys)
        created_keys = created_stats.select(keys).unique().sort(keys)
        missing_units = keys_df.join(created_keys, on=keys, how="anti").sort(keys)
        self.stats["missing_units_n"] = int(missing_units.height)
        self.stats["missing_units_df"] = missing_units

        bounds_df = bounds_lf.select(
            keys + ["has_overlap", "start_frame_req", "end_frame_req", "min_raw", "max_raw", "start_frame", "end_frame"]
        ).collect(engine="streaming")

        missing_detail = (
            missing_units.join(bounds_df, on=keys, how="left")
            .with_columns(
                pl.when(pl.col("min_raw").is_null())
                .then(pl.lit("no_raw_rows"))
                .when(pl.col("has_overlap") == False)
                .then(pl.lit("no_overlap"))
                .otherwise(pl.lit("unknown"))
                .alias("reason")
            )
            .select(keys + ["reason", "start_frame_req", "end_frame_req", "min_raw", "max_raw"])
            .sort(keys)
        )
        self.stats["missing_units_detail_df"] = missing_detail

        if missing_detail.height > 0:
            head_n = min(20, int(missing_detail.height))
            lines = []
            for rr in missing_detail.head(head_n).iter_rows(named=True):
                lines.append(
                    f"- subject={rr['subject']}, velocity={rr['velocity']}, trial_num={rr['trial_num']}, reason={rr['reason']}"
                )
            logger.warning(
                f"Stage01 세그먼트 누락 unit: {missing_detail.height}개 (상위 {head_n}개 표시)\n"
                + "\n".join(lines)
            )

        # Drop helper columns not present in legacy outputs
        helper_cols = [
            "start_frame_req",
            "end_frame_req",
            "start_frame",
            "end_frame",
            "plat_order",
            "min_raw",
            "max_raw",
            "has_overlap",
        ]
        segmented_cols = segmented.collect_schema().names()
        helper_present = [c for c in helper_cols if c in segmented_cols]
        if helper_present:
            segmented = segmented.drop(helper_present)

        logger.info(
            f"Dataset built (lazy): {self.stats.get('total_rows', 'unknown')} rows, "
            f"{len(segmented.collect_schema().names())} columns, "
            f"{self.stats.get('segments_created', 'unknown')} segmented trials"
        )

        return segmented

    def _validate_output_lazy(self, lf: pl.LazyFrame) -> pl.DataFrame:
        """Validations that avoid collecting the full dataset; returns per-unit stats."""
        keys = ["subject", "velocity", "trial_num"]
        required = keys + ["original_DeviceFrame", "DeviceFrame", "start_frame"]
        lf_cols = lf.collect_schema().names()
        missing = [c for c in required if c not in lf_cols]
        if missing:
            raise ValueError(f"Stage01 validation failed: missing required columns: {missing}")

        stats_lf = lf.group_by(keys).agg([
            pl.len().cast(pl.Int64).alias("n_rows"),
            pl.col("original_DeviceFrame").cast(pl.Int64).min().alias("odf_min"),
            pl.col("original_DeviceFrame").cast(pl.Int64).max().alias("odf_max"),
            pl.col("original_DeviceFrame").cast(pl.Int64).sum().alias("odf_sum"),
            (pl.col("original_DeviceFrame").cast(pl.Int64) * pl.col("original_DeviceFrame").cast(pl.Int64))
            .sum()
            .alias("odf_sumsq"),
            pl.col("DeviceFrame").cast(pl.Int64).min().alias("df_min"),
            pl.col("DeviceFrame").cast(pl.Int64).max().alias("df_max"),
            pl.col("DeviceFrame").cast(pl.Int64).sum().cast(pl.Int64).alias("df_sum"),
            (pl.col("DeviceFrame").cast(pl.Int64) * pl.col("DeviceFrame").cast(pl.Int64))
            .sum()
            .cast(pl.Int64)
            .alias("df_sumsq"),
        ])

        stats = stats_lf.collect(engine="streaming")
        if stats.is_empty():
            raise ValueError("No segments were successfully created")

        self.stats["segments_created"] = int(stats.height)
        self.stats["total_rows"] = int(stats.get_column("n_rows").sum())

        # Expected values for 0..n-1 sequence
        expected_sum = (
            pl.col("n_rows") * (pl.col("n_rows") - 1) // 2
        ).cast(pl.Int64)
        expected_sumsq = (
            (pl.col("n_rows") - 1)
            * pl.col("n_rows")
            * (2 * pl.col("n_rows") - 1)
            // 6
        ).cast(pl.Int64)

        # Expected sums for shifted original_DeviceFrame (x - min) where x should be consecutive
        odf_expected_shift_sum = expected_sum
        odf_expected_shift_sumsq = expected_sumsq
        odf_shift_sum = (pl.col("odf_sum") - pl.col("n_rows") * pl.col("odf_min")).cast(pl.Int64)
        odf_shift_sumsq = (
            pl.col("odf_sumsq")
            - 2 * pl.col("odf_min") * pl.col("odf_sum")
            + pl.col("n_rows") * pl.col("odf_min") * pl.col("odf_min")
        ).cast(pl.Int64)

        bad = stats.filter(
            # original_DeviceFrame must be consecutive within each segment
            (pl.col("odf_max") - pl.col("odf_min") != pl.col("n_rows") - 1)
            | (odf_shift_sum != odf_expected_shift_sum)
            | (odf_shift_sumsq != odf_expected_shift_sumsq)
            # DeviceFrame must be 0..n-1
            | (pl.col("df_min") != 0)
            | (pl.col("df_max") != pl.col("n_rows") - 1)
            | (pl.col("df_sum") != expected_sum)
            | (pl.col("df_sumsq") != expected_sumsq)
        )

        if bad.height > 0:
            raise ValueError(
                f"Stage01 validation failed: non-consecutive frames in {bad.height} trials"
            )
        return stats

    def merge_platform_metadata_lazy(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """LazyFrame variant of platform metadata merge (streamable)."""
        logger.info("Merging platform metadata (lazy)...")

        if self.platform_metadata_df is None:
            logger.warning("No platform metadata available for merge")
            return lf

        metadata = self.platform_metadata_df.clone()
        if "trial" in metadata.columns and "trial_num" not in metadata.columns:
            metadata = metadata.rename({"trial": "trial_num"})

        meta_lf = metadata.lazy().with_columns([
            pl.col("subject").cast(pl.Utf8),
            pl.col("velocity").cast(pl.Float64, strict=False),
            pl.col("trial_num").cast(pl.Int64, strict=False),
        ])

        merged = lf.join(
            meta_lf,
            on=["subject", "velocity", "trial_num"],
            how="left",
            suffix="_meta",
        )

        dup_cols = [c for c in merged.collect_schema().names() if c.endswith("_meta")]
        if dup_cols:
            merged = merged.drop(dup_cols)

        return merged

    def _resolve_stage1_corrections_csv_path_for_output(self, *, output_dir: Path) -> Path:
        s1_cfg = (self.config.get("forceplate", {}) or {}).get("stage1_corrections", {}) or {}
        corr_path = s1_cfg.get("corrections_csv")
        if corr_path:
            corr_csv = Path(str(corr_path))
            if not corr_csv.is_absolute():
                corr_csv = (MODULE_DIR / corr_csv).resolve()
            return corr_csv

        rel = (
            (self.config.get("pipeline_files", {}) or {}).get("stage01_forceplate_stage1_corrections_csv")
            or "stage1/corrections.csv"
        )
        return (Path(output_dir) / rel).resolve()

    def _resolve_stage1_report_md_path_for_output(self, *, output_dir: Path) -> Path:
        rel = (
            (self.config.get("pipeline_files", {}) or {}).get("stage01_forceplate_stage1_report_md")
            or "stage1/report.md"
        )
        return (Path(output_dir) / rel).resolve()

    def ensure_forceplate_stage1_corrections(self, *, output_dir: Path) -> None:
        """
        Ensure Stage1(window correction) correction table exists.

        Notes:
        - This generates corrections.csv inside this repo (no external dependency).
        - This must NOT change segmentation.pre_frames/post_frames behavior; it only computes shift values.
        """
        s1_cfg = (self.config.get("forceplate", {}) or {}).get("stage1_corrections", {}) or {}
        if not bool(s1_cfg.get("enabled", False)):
            return

        corr_csv = self._resolve_stage1_corrections_csv_path_for_output(output_dir=output_dir)
        if corr_csv.exists():
            logger.info(f"Stage1 corrections_csv already exists (skip): {corr_csv}")
            return

        self._generate_forceplate_stage1_corrections(
            corr_csv=corr_csv,
            report_md=self._resolve_stage1_report_md_path_for_output(output_dir=output_dir),
        )

    def _generate_forceplate_stage1_corrections(self, *, corr_csv: Path, report_md: Path) -> pl.DataFrame:
        """
        Compute Stage1 shift corrections (BW_ref 기반 window correction) and save to CSV.

        Implementation mirrors Archive/codebase.xml Stage1 logic, but runs within this repo and
        intentionally does not alter Stage01 segmentation output ranges.
        """
        mocap_hz = float((self.config.get("sampling", {}) or {}).get("mocap_hz", 100))
        stage1_cfg = self.config.get("stage1", {}) or {}
        baseline_cfg = stage1_cfg.get("baseline", {}) or {}
        thr_cfg = stage1_cfg.get("thresholds", {}) or {}
        window_cfg = stage1_cfg.get("analysis_window_sec", {}) or {}

        baseline_sec = float(baseline_cfg.get("window_sec", 1.0))
        baseline_frames = int(round(baseline_sec * mocap_hz))
        baseline_min_rows = int(baseline_cfg.get("min_rows", 500))
        inclusive_onset = bool(baseline_cfg.get("inclusive_onset", True))

        fz_low_thr_n = float(thr_cfg.get("fz_low_thr_n", 20.0))
        fz_high_thr_n = float(thr_cfg.get("fz_high_thr_n", 200.0))
        loaded_high_frac_min = float(thr_cfg.get("loaded_high_frac_min", 0.1))
        moment_abs_max_min_nm = float(thr_cfg.get("moment_abs_max_min_nm", 10.0))
        min_abs_shift_to_apply_n = float(thr_cfg.get("min_abs_shift_to_apply_n", 100.0))
        good_high_frac_min = float(thr_cfg.get("good_high_frac_min", 0.5))

        pre_sec = float(window_cfg.get("pre_onset", 1.0))
        post_sec = float(window_cfg.get("post_offset", 2.0))
        pre_frames = int(round(pre_sec * mocap_hz))
        post_frames = int(round(post_sec * mocap_hz))

        keys = ["subject", "velocity_key", "trial_num"]

        plat = self.platform_timing_df.clone()
        if "trial_num" not in plat.columns and "trial" in plat.columns:
            plat = plat.rename({"trial": "trial_num"})

        required_plat = {"subject", "velocity", "trial_num", "platform_onset", "platform_offset"}
        missing_plat = sorted(required_plat - set(plat.columns))
        if missing_plat:
            raise ValueError(f"Stage1 corrections 생성에 필요한 platform sheet 컬럼이 없습니다: {missing_plat}")

        plat = (
            plat.with_columns(
                [
                    pl.col("subject").cast(pl.Utf8).alias("subject"),
                    pl.col("velocity").cast(pl.Float64, strict=False).round(6).alias("velocity_key"),
                    pl.col("trial_num").cast(pl.Int64, strict=False).alias("trial_num"),
                    pl.col("platform_onset").cast(pl.Int64, strict=False).alias("onset"),
                    pl.col("platform_offset").cast(pl.Int64, strict=False).alias("offset"),
                ]
            )
            .filter(pl.col("onset").is_not_null() & pl.col("offset").is_not_null())
            .unique(subset=["subject", "velocity_key", "trial_num"], keep="first")
        )

        lf_cols = self.parquet_lf.collect_schema().names()
        needed = ["DeviceFrame", "MocapFrame", "Fx", "Fy", "Fz", "Mx", "My", "Mz", "subject", "velocity", "trial_num"]
        cols = [c for c in needed if c in lf_cols]
        if "MocapFrame" not in cols or "Fz" not in cols:
            raise ValueError("Stage1 corrections 생성에 필요한 MocapFrame/Fz 컬럼이 원본 parquet에 없습니다.")

        raw = (
            self.parquet_lf.select(cols)
            .with_columns(
                [
                    pl.col("subject").cast(pl.Utf8).alias("subject"),
                    pl.col("velocity").cast(pl.Float64, strict=False).round(6).alias("velocity_key"),
                    pl.col("trial_num").cast(pl.Int64, strict=False).alias("trial_num"),
                    pl.col("MocapFrame").cast(pl.Int64, strict=False).alias("MocapFrame"),
                ]
            )
            .join(plat.lazy(), on=keys, how="inner")
        )

        base_end = pl.col("onset") if inclusive_onset else (pl.col("onset") - pl.lit(1))
        base_start = pl.max_horizontal(pl.lit(0), pl.col("onset") - pl.lit(int(baseline_frames)))
        baseline = raw.filter((pl.col("MocapFrame") >= base_start) & (pl.col("MocapFrame") <= base_end))

        def _maybe_median(col: str, alias: str) -> pl.Expr:
            if col in cols:
                return pl.col(col).cast(pl.Float64, strict=False).median().alias(alias)
            return pl.lit(None).cast(pl.Float64).alias(alias)

        def _maybe_absmax(col: str, alias: str) -> pl.Expr:
            if col in cols:
                return pl.col(col).cast(pl.Float64, strict=False).abs().max().alias(alias)
            return pl.lit(None).cast(pl.Float64).alias(alias)

        high = pl.col("Fz").cast(pl.Float64, strict=False).abs() > pl.lit(float(fz_high_thr_n))
        baseline_stats = (
            baseline.group_by(keys)
            .agg(
                [
                    pl.len().alias("baseline_n_rows"),
                    high.cast(pl.Float64).mean().alias("baseline_fz_high_frac"),
                    pl.col("Fz").cast(pl.Float64, strict=False).median().alias("baseline_fz_median"),
                    pl.when(high).then(pl.col("Fz").cast(pl.Float64, strict=False)).otherwise(None).median().alias("baseline_fz_high_median"),
                    _maybe_median("Fx", "baseline_fx_median"),
                    _maybe_median("Fy", "baseline_fy_median"),
                    _maybe_median("Mx", "baseline_mx_median"),
                    _maybe_median("My", "baseline_my_median"),
                    _maybe_median("Mz", "baseline_mz_median"),
                    _maybe_absmax("Mx", "baseline_mx_abs_max"),
                    _maybe_absmax("My", "baseline_my_abs_max"),
                ]
            )
            .join(plat.lazy(), on=keys, how="left")
        )

        good = baseline_stats.filter(
            (pl.col("baseline_n_rows") >= pl.lit(int(baseline_min_rows)))
            & (pl.col("baseline_fz_high_frac") >= pl.lit(float(good_high_frac_min)))
            & (pl.col("baseline_fz_high_median").abs() >= pl.lit(float(fz_high_thr_n)))
        )

        bw_ref_by_subject = good.group_by("subject").agg(pl.col("baseline_fz_high_median").median().alias("bw_ref_n"))
        global_bw_ref = float("nan")
        try:
            gb = good.select(pl.col("baseline_fz_high_median").median()).collect(engine="streaming").to_series()
            if len(gb) == 1:
                global_bw_ref = float(gb[0])
        except Exception:
            global_bw_ref = float("nan")

        merged = baseline_stats.join(bw_ref_by_subject, on="subject", how="left").with_columns(
            pl.when(pl.col("bw_ref_n").is_null()).then(pl.lit(global_bw_ref)).otherwise(pl.col("bw_ref_n")).alias("bw_ref_n")
        )

        moment_abs = pl.max_horizontal(
            pl.col("baseline_mx_abs_max").fill_null(0.0).abs(),
            pl.col("baseline_my_abs_max").fill_null(0.0).abs(),
        )
        shift_fz = pl.col("bw_ref_n") - pl.col("baseline_fz_median")
        apply = (
            pl.col("bw_ref_n").is_not_null()
            & (pl.col("baseline_n_rows") >= pl.lit(int(baseline_min_rows)))
            & (pl.col("baseline_fz_high_frac") < pl.lit(float(loaded_high_frac_min)))
            & (moment_abs >= pl.lit(float(moment_abs_max_min_nm)))
            & (shift_fz.abs() >= pl.lit(float(min_abs_shift_to_apply_n)))
        ).fill_null(False)

        out = merged.with_columns(
            [
                apply.alias("correction_applied"),
                pl.when(apply).then(shift_fz).otherwise(pl.lit(0.0)).cast(pl.Float64).alias("shift_n"),
                pl.when(apply).then(-pl.col("baseline_fx_median")).otherwise(pl.lit(0.0)).cast(pl.Float64).alias("shift_fx_n"),
                pl.when(apply).then(-pl.col("baseline_fy_median")).otherwise(pl.lit(0.0)).cast(pl.Float64).alias("shift_fy_n"),
                pl.when(apply).then(-pl.col("baseline_mx_median")).otherwise(pl.lit(0.0)).cast(pl.Float64).alias("shift_mx_nm"),
                pl.when(apply).then(-pl.col("baseline_my_median")).otherwise(pl.lit(0.0)).cast(pl.Float64).alias("shift_my_nm"),
                pl.when(apply).then(-pl.col("baseline_mz_median")).otherwise(pl.lit(0.0)).cast(pl.Float64).alias("shift_mz_nm"),
                pl.max_horizontal(pl.lit(0), pl.col("onset") - pl.lit(int(pre_frames))).cast(pl.Int64).alias("window_start"),
                (pl.col("offset") + pl.lit(int(post_frames))).cast(pl.Int64).alias("window_end"),
                ((pl.col("offset") + pl.lit(int(post_frames)) - pl.max_horizontal(pl.lit(0), pl.col("onset") - pl.lit(int(pre_frames)))) / pl.lit(float(mocap_hz)))
                .cast(pl.Float64)
                .alias("window_sec"),
                pl.col("velocity_key").cast(pl.Float64).alias("velocity"),
            ]
        ).select(
            [
                "subject",
                "velocity",
                "trial_num",
                "onset",
                "offset",
                "window_start",
                "window_end",
                "window_sec",
                "bw_ref_n",
                "baseline_n_rows",
                "baseline_fz_high_frac",
                "baseline_fz_median",
                "baseline_fz_high_median",
                "baseline_fx_median",
                "baseline_fy_median",
                "baseline_mx_median",
                "baseline_my_median",
                "baseline_mz_median",
                "baseline_mx_abs_max",
                "baseline_my_abs_max",
                "correction_applied",
                "shift_n",
                "shift_fx_n",
                "shift_fy_n",
                "shift_mx_nm",
                "shift_my_nm",
                "shift_mz_nm",
            ]
        )

        out_df = out.sort(["subject", "velocity", "trial_num"]).collect(engine="streaming")
        corr_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.write_csv(corr_csv, include_header=True)

        n_total = int(out_df.height)
        n_applied = int(out_df.select((pl.col("correction_applied") == True).sum()).item())
        lines: list[str] = []
        lines.append("# Stage1: BW_ref 기반 window 보정 요약")
        lines.append("")
        lines.append(f"- 출력(corrections): `{corr_csv}`")
        lines.append("")
        lines.append("## 설정")
        lines.append(f"- preonset baseline window: {baseline_sec:g}s (MocapFrame 기준)")
        lines.append(f"- window: [onset-{pre_sec:g}s, offset+{post_sec:g}s] (MocapFrame 기준)")
        lines.append(f"- loaded 판정(high_frac): < {loaded_high_frac_min:g}")
        lines.append(f"- moment abs max(min): >= {moment_abs_max_min_nm:g} Nm")
        lines.append(f"- min abs shift to apply: >= {min_abs_shift_to_apply_n:g} N")
        lines.append("")
        lines.append("## 결과")
        lines.append(f"- unit 수: {n_total}")
        lines.append(f"- 보정 적용 unit 수: {n_applied}")
        lines.append("")
        if n_applied:
            top = (
                out_df.filter(pl.col("correction_applied") == True)
                .with_columns(pl.col("shift_n").abs().alias("abs_shift_n"))
                .sort("abs_shift_n", descending=True)
                .head(30)
            )
            lines.append("## 보정 적용 Top 30 (|shift| 기준)")
            lines.append("| subject | velocity | trial | onset | offset | shift(N) | baseline_fz_median | bw_ref_n |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for rr in top.iter_rows(named=True):
                lines.append(
                    f"| {rr['subject']} | {float(rr['velocity']):g} | {int(rr['trial_num'])} | {int(rr['onset'])} | {int(rr['offset'])} | {float(rr['shift_n']):.1f} | {float(rr['baseline_fz_median']):.1f} | {float(rr['bw_ref_n']):.1f} |"
                )
            lines.append("")

        report_md.parent.mkdir(parents=True, exist_ok=True)
        report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(f"Stage1 corrections 생성 완료: {corr_csv}")
        return out_df

    def apply_forceplate_stage1_corrections_lazy(self, lf: pl.LazyFrame, *, output_dir: Path) -> pl.LazyFrame:
        """
        Apply stage1(window correction) shift offsets to forceplate channels.

        This aligns Stage01 forceplate signals with Archive/codebase.xml semantics where
        Fz_corrected reflects absolute BW (via shift_n) and COP uses corrected Fz/Mx/My.
        """
        s1_cfg = (self.config.get("forceplate", {}) or {}).get("stage1_corrections", {}) or {}
        if not bool(s1_cfg.get("enabled", False)):
            logger.info("Stage1 shift 보정(forceplate.stage1_corrections.enabled=false): skip")
            return lf

        corr_csv = self._resolve_stage1_corrections_csv_path_for_output(output_dir=output_dir)
        if not corr_csv.exists():
            raise FileNotFoundError(f"stage1 corrections_csv not found: {corr_csv}")

        columns_map = s1_cfg.get("columns", {}) or {}
        if not isinstance(columns_map, dict) or not columns_map:
            raise ValueError("forceplate.stage1_corrections.columns 설정이 비어있습니다.")

        ds_cols = lf.collect_schema().names()
        target_channels = [ch for ch in columns_map.keys() if ch in ds_cols]
        if not target_channels:
            logger.warning("Stage1 shift 보정: 대상 forceplate 채널이 Stage01 데이터에 없습니다 (skip)")
            return lf

        shift_cols = sorted({str(v) for v in columns_map.values()})
        required_cols = {"subject", "velocity", "trial_num", "correction_applied"} | set(shift_cols)
        keep_cols = ["subject", "trial_num", "correction_applied", "velocity_key"] + shift_cols

        corr = pl.read_csv(corr_csv)
        missing = sorted(required_cols - set(corr.columns))
        if missing:
            raise ValueError(f"stage1 corrections_csv에 필요한 컬럼이 없습니다: {missing}")

        corr = (
            corr.with_columns(
                [
                    pl.col("subject").cast(pl.Utf8).str.strip_chars().alias("subject"),
                    pl.col("velocity").cast(pl.Float64, strict=False).round(6).alias("velocity_key"),
                    pl.col("trial_num").cast(pl.Int64, strict=False).alias("trial_num"),
                    pl.col("correction_applied").cast(pl.Boolean, strict=False).alias("correction_applied"),
                ]
            )
            .with_columns([pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in shift_cols])
            .select(keep_cols)
            .unique(subset=["subject", "velocity_key", "trial_num"], keep="first")
        )

        apply_only_if_applied = bool(s1_cfg.get("apply_only_if_correction_applied", True))
        if apply_only_if_applied:
            apply_expr = pl.col("correction_applied") == True
        else:
            any_shift = [pl.col(c).is_not_null() for c in shift_cols]
            apply_expr = pl.any_horizontal(any_shift) if any_shift else pl.lit(False)

        joined = (
            lf.with_columns(pl.col("velocity").cast(pl.Float64, strict=False).round(6).alias("velocity_key"))
            .join(corr.lazy(), on=["subject", "velocity_key", "trial_num"], how="left")
        )

        updates: list[pl.Expr] = []
        for ch in target_channels:
            shift_col = str(columns_map[ch])
            if shift_col not in joined.collect_schema().names():
                continue
            updates.append(
                pl.when(apply_expr & pl.col(shift_col).is_not_null())
                .then(pl.col(ch).cast(pl.Float64, strict=False) + pl.col(shift_col).cast(pl.Float64, strict=False))
                .otherwise(pl.col(ch).cast(pl.Float64, strict=False))
                .alias(ch)
            )

        if updates:
            joined = joined.with_columns(updates)

        drop_cols = [c for c in (["velocity_key", "correction_applied"] + shift_cols) if c in joined.collect_schema().names()]
        if drop_cols:
            joined = joined.drop(drop_cols)

        logger.info(f"Stage1 shift 보정 적용 완료: {corr_csv}")
        return joined

    def _build_forceplate_100hz_trial_lists(self, dataset_lf: pl.LazyFrame) -> pl.DataFrame:
        keys = ["subject", "velocity", "trial_num"]
        frame_ratio = get_frame_ratio(self.config)

        required = ["DeviceFrame", "MocapFrame", "Fx", "Fy", "Fz", "Mx", "My", "Mz", "platform_onset", "platform_offset"]
        ds_cols = dataset_lf.collect_schema().names()
        missing = [c for c in required if c not in ds_cols]
        if missing:
            raise ValueError(f"Forceplate 관성 성분 제거에 필요한 컬럼이 Stage01 데이터에 없습니다: {missing}")

        fp100 = (
            dataset_lf.with_columns(
                (pl.col("DeviceFrame") // pl.lit(frame_ratio)).cast(pl.Int64).alias("mocap_idx_local")
            )
            .group_by(keys + ["mocap_idx_local"])
            .agg(
                [
                    pl.col("MocapFrame").cast(pl.Int64).first().alias("MocapFrame"),
                    pl.col("Fx").cast(pl.Float64, strict=False).mean().alias("Fx"),
                    pl.col("Fy").cast(pl.Float64, strict=False).mean().alias("Fy"),
                    pl.col("Mx").cast(pl.Float64, strict=False).mean().alias("Mx"),
                    pl.col("My").cast(pl.Float64, strict=False).mean().alias("My"),
                    pl.col("Mz").cast(pl.Float64, strict=False).mean().alias("Mz"),
                    pl.col("Fz").cast(pl.Float64, strict=False).mean().alias("Fz"),
                    pl.col("platform_onset").cast(pl.Int64, strict=False).first().alias("platform_onset"),
                    pl.col("platform_offset").cast(pl.Int64, strict=False).first().alias("platform_offset"),
                ]
            )
            .group_by(keys)
            .agg(
                [
                    pl.col("mocap_idx_local").sort().alias("mocap_idx_local"),
                    pl.col("MocapFrame").sort_by("mocap_idx_local").alias("MocapFrame"),
                    pl.col("Fx").sort_by("mocap_idx_local").alias("Fx"),
                    pl.col("Fy").sort_by("mocap_idx_local").alias("Fy"),
                    pl.col("Mx").sort_by("mocap_idx_local").alias("Mx"),
                    pl.col("My").sort_by("mocap_idx_local").alias("My"),
                    pl.col("Mz").sort_by("mocap_idx_local").alias("Mz"),
                    pl.col("Fz").sort_by("mocap_idx_local").alias("Fz"),
                    pl.first("platform_onset").alias("platform_onset"),
                    pl.first("platform_offset").alias("platform_offset"),
                ]
            )
        )
        return fp100.collect(engine="streaming")

    def apply_forceplate_inertial_removal_lazy(self, dataset_lf: pl.LazyFrame, output_dir: Path) -> pl.LazyFrame:
        fp_cfg = self.config.get("forceplate", {}) or {}
        ir_cfg = fp_cfg.get("inertial_removal", {}) or {}
        if not bool(ir_cfg.get("enabled", False)):
            logger.info("Forceplate 관성 성분 제거(forceplate.inertial_removal.enabled=false): skip")
            return dataset_lf

        return self.apply_forceplate_subtract_lazy(dataset_lf, output_dir=output_dir)

    def apply_forceplate_zeroed_columns_lazy(self, dataset_lf: pl.LazyFrame) -> pl.LazyFrame:
        fp_cfg = self.config.get("forceplate", {}) or {}
        zero_cfg = fp_cfg.get("zeroed", {}) or {}
        if not bool(zero_cfg.get("enabled", False)):
            logger.info("Forceplate zeroed columns 생성(forceplate.zeroed.enabled=false): skip")
            return dataset_lf

        keys = ["subject", "velocity", "trial_num"]
        pre_frames = int(self.pre_frames)

        ds_cols = dataset_lf.collect_schema().names()
        required = ["original_DeviceFrame"]
        missing = [c for c in required if c not in ds_cols]
        if missing:
            logger.warning(f"Forceplate zeroed columns skip (missing columns): {missing}")
            return dataset_lf

        cols_cfg = zero_cfg.get("columns")
        if cols_cfg is None:
            zero_cols = list(fp_cfg.get("merge_columns", []) or [])
        elif isinstance(cols_cfg, list):
            zero_cols = [str(c).strip() for c in cols_cfg]
        else:
            raise ValueError("forceplate.zeroed.columns는 list[str] 이어야 합니다.")

        if not zero_cols:
            zero_cols = list(fp_cfg.get("merge_columns", []) or [])

        zero_cols = [c for c in zero_cols if c in ds_cols]
        if not zero_cols:
            logger.warning("Forceplate zeroed columns skip (no matching columns in dataset)")
            return dataset_lf

        suffix = str(zero_cfg.get("suffix", "_zero"))
        base_prefix = "_fp_zero_base_"

        with_start = dataset_lf.with_columns(
            pl.col("original_DeviceFrame").min().over(keys).alias("_fp_zero_start")
        )
        mask = (pl.col("original_DeviceFrame") >= pl.col("_fp_zero_start")) & (
            pl.col("original_DeviceFrame") < (pl.col("_fp_zero_start") + pl.lit(pre_frames))
        )

        baseline_exprs: list[pl.Expr] = []
        for c in zero_cols:
            baseline_exprs.append(
                pl.col(c)
                .cast(pl.Float64, strict=False)
                .filter(mask)
                .mean()
                .alias(f"{base_prefix}{c}")
            )

        baselines = with_start.group_by(keys).agg(baseline_exprs)
        joined = with_start.join(baselines, on=keys, how="left")

        updates: list[pl.Expr] = []
        for c in zero_cols:
            base_c = f"{base_prefix}{c}"
            updates.append(
                pl.when(pl.col(base_c).is_not_null())
                .then(pl.col(c).cast(pl.Float64, strict=False) - pl.col(base_c))
                .otherwise(pl.lit(None).cast(pl.Float64))
                .alias(f"{c}{suffix}")
            )

        drop_cols = [f"{base_prefix}{c}" for c in zero_cols] + ["_fp_zero_start"]
        return joined.with_columns(updates).drop(drop_cols)

    def _compute_forceplate_fz_qc_summary(self, diag_path: Path) -> pl.DataFrame:
        stage1_thr = ((self.config.get("stage1", {}) or {}).get("thresholds", {}) or {})
        threshold_n = float(stage1_thr.get("fz_high_thr_n", 392.3))

        quantile_p = 0.05

        lf = pl.scan_parquet(diag_path)
        required = {
            "subject",
            "velocity",
            "trial_num",
            "mocap_idx_local",
            "MocapFrame",
            "onset_local_100hz",
            "offset_local_100hz",
            "Fz_measured_100hz",
            "Fz_corrected_100hz",
        }
        schema = lf.collect_schema().names()
        missing = sorted(required - set(schema))
        if missing:
            raise ValueError(f"diagnostics parquet missing required columns: {missing}")

        keys = ["subject", "velocity", "trial_num"]
        null_counts = lf.select([pl.col(k).null_count().alias(k) for k in keys]).collect(engine="streaming")
        null_info = {k: int(null_counts[k][0]) for k in keys}
        if any(v > 0 for v in null_info.values()):
            raise ValueError(f"subject/velocity/trial_num contains nulls: {null_info}")

        moment_cols: list[str] = []
        if "Mx_measured_100hz" in schema:
            moment_cols.append("Mx_measured_100hz")
        if "My_measured_100hz" in schema:
            moment_cols.append("My_measured_100hz")

        select_cols = (
            keys
            + [
                "mocap_idx_local",
                "MocapFrame",
                "onset_local_100hz",
                "offset_local_100hz",
                "Fz_measured_100hz",
                "Fz_corrected_100hz",
            ]
            + moment_cols
        )

        lf = (
            lf.select(select_cols)
            .with_columns(
                [
                    pl.col("subject").cast(pl.Utf8).str.strip_chars().alias("subject"),
                    pl.col("velocity").cast(pl.Float64, strict=False).alias("velocity"),
                    pl.col("trial_num").cast(pl.Int64, strict=False).alias("trial_num"),
                    pl.col("mocap_idx_local").cast(pl.Int64, strict=False).alias("mocap_idx_local"),
                    pl.col("MocapFrame").cast(pl.Int64, strict=False).alias("MocapFrame"),
                    pl.col("onset_local_100hz").cast(pl.Int64, strict=False).alias("onset_local_100hz"),
                    pl.col("offset_local_100hz").cast(pl.Int64, strict=False).alias("offset_local_100hz"),
                ]
            )
            .sort(keys + ["mocap_idx_local"])
            .with_columns(pl.int_range(0, pl.len()).over(keys).cast(pl.Int64).alias("row_idx"))
        )

        base = lf.group_by(keys).agg(
            [
                pl.len().alias("n_rows"),
                pl.col("mocap_idx_local").n_unique().alias("n_time_unique"),
                pl.col("mocap_idx_local").null_count().alias("n_time_null"),
                pl.col("MocapFrame").null_count().alias("n_mocap_null"),
                pl.col("onset_local_100hz").drop_nulls().first().alias("onset_local_100hz"),
                pl.col("offset_local_100hz").drop_nulls().first().alias("offset_local_100hz"),
                pl.col("MocapFrame")
                .filter(pl.col("row_idx") == pl.col("onset_local_100hz"))
                .first()
                .alias("onset_mocap_frame"),
                pl.col("MocapFrame")
                .filter(pl.col("row_idx") == pl.col("offset_local_100hz"))
                .first()
                .alias("offset_mocap_frame"),
                pl.col("mocap_idx_local").diff().min().alias("mocap_idx_step_min"),
                pl.col("MocapFrame").diff().min().alias("mocap_frame_step_min"),
            ]
        )

        window_lf = lf.filter(
            (pl.col("row_idx") >= pl.col("onset_local_100hz"))
            & (pl.col("row_idx") <= pl.col("offset_local_100hz"))
        )

        def _metric_exprs(col: str, prefix: str) -> list[pl.Expr]:
            abs_col = pl.col(col).cast(pl.Float64, strict=False).abs()
            valid = abs_col.is_finite()
            return [
                abs_col.filter(valid).median().alias(f"{prefix}_median_n"),
                abs_col.filter(valid).quantile(float(quantile_p), "nearest").alias(f"{prefix}_p05_n"),
                (abs_col < pl.lit(float(threshold_n))).filter(valid).mean().alias(f"{prefix}_below_frac"),
            ]

        aggs: list[pl.Expr] = [pl.len().alias("window_n_rows")]
        aggs += _metric_exprs("Fz_measured_100hz", "fz_measured")
        aggs += _metric_exprs("Fz_corrected_100hz", "fz_corrected")

        if {"Mx_measured_100hz", "My_measured_100hz"}.issubset(set(schema)):
            moment_abs = pl.max_horizontal(
                pl.col("Mx_measured_100hz").cast(pl.Float64, strict=False).abs(),
                pl.col("My_measured_100hz").cast(pl.Float64, strict=False).abs(),
            )
            aggs.append(moment_abs.filter(moment_abs.is_finite()).max().alias("moment_abs_max_nm"))

        window_stats = window_lf.group_by(keys).agg(aggs)

        summary = base.join(window_stats, on=keys, how="left").with_columns(
            [
                pl.col("window_n_rows").fill_null(0).cast(pl.Int64).alias("window_n_rows"),
                pl.col("n_rows").cast(pl.Int64).alias("n_rows"),
                pl.col("n_time_unique").cast(pl.Int64).alias("n_time_unique"),
                (
                    pl.col("onset_local_100hz").is_not_null()
                    & (pl.col("onset_local_100hz") >= 0)
                    & (pl.col("onset_local_100hz") < pl.col("n_rows"))
                ).alias("onset_in_range"),
                (
                    pl.col("offset_local_100hz").is_not_null()
                    & (pl.col("offset_local_100hz") >= 0)
                    & (pl.col("offset_local_100hz") < pl.col("n_rows"))
                ).alias("offset_in_range"),
                (
                    pl.col("offset_local_100hz").is_not_null()
                    & pl.col("onset_local_100hz").is_not_null()
                    & (pl.col("offset_local_100hz") >= pl.col("onset_local_100hz"))
                ).alias("offset_after_onset"),
            ]
        ).with_columns(
            [
                (pl.col("n_time_unique") < pl.col("n_rows")).alias("time_index_has_duplicates"),
                (pl.col("mocap_frame_step_min") < 0).alias("mocap_frame_nonmonotonic"),
                (pl.col("mocap_idx_step_min") < 0).alias("mocap_idx_nonmonotonic"),
                (
                    pl.col("onset_in_range")
                    & pl.col("offset_in_range")
                    & pl.col("offset_after_onset")
                    & (pl.col("window_n_rows") > 0)
                ).alias("window_valid"),
                pl.lit(float(threshold_n)).cast(pl.Float64).alias("fz_threshold_n"),
            ]
        )

        return summary.sort(keys).collect(engine="streaming")

    def write_forceplate_fz_qc_summary_csv(self, *, diag_path: Path, output_dir: Path) -> None:
        qc_rel = (
            (self.config.get("pipeline_files", {}) or {}).get("stage01_forceplate_cop_qc_dir")
            or "cop_qc"
        )
        out_dir = (Path(output_dir) / qc_rel).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        fz_qc_path = out_dir / "fz_qc_summary.csv"

        summary = self._compute_forceplate_fz_qc_summary(diag_path)
        summary.write_csv(fz_qc_path, include_header=True)
        logger.info(f"Fz QC summary 저장: {fz_qc_path}")

    def apply_forceplate_subtract_lazy(self, dataset_lf: pl.LazyFrame, *, output_dir: Path) -> pl.LazyFrame:
        fp_cfg = self.config.get("forceplate", {}) or {}
        sub_cfg = fp_cfg.get("subtract", {}) or {}

        frame_ratio = get_frame_ratio(self.config)
        keys = ["subject", "velocity", "trial_num"]

        apply_channels_raw = sub_cfg.get("apply_channels")
        if apply_channels_raw is None:
            apply_channels = ["Fx", "My", "Mz"]
        elif isinstance(apply_channels_raw, list):
            apply_channels = [str(x).strip() for x in apply_channels_raw]
        else:
            raise ValueError("forceplate.subtract.apply_channels는 list[str] 이어야 합니다.")

        supported_channels = {"Fx", "Fy", "Fz", "Mx", "My", "Mz"}
        unknown = [c for c in apply_channels if c not in supported_channels]
        if unknown:
            raise ValueError(f"forceplate.subtract.apply_channels에 지원하지 않는 채널이 있습니다: {unknown}")
        if not apply_channels:
            raise ValueError("forceplate.subtract.apply_channels가 비어있습니다.")

        templates = self._try_build_unloaded_velocity_templates_subtract()

        logger.info("Direct subtract: 100Hz 진단 테이블 생성 중...")
        trial_lists = self._build_forceplate_100hz_trial_lists(dataset_lf)
        diagnostics = self._compute_subtract_diagnostics_from_trial_lists(
            trial_lists,
            templates=templates,
        )

        diag_name = (
            self.config.get("pipeline_files", {}).get("stage01_forceplate_subtract_diagnostics")
            or "forceplate_subtract_diagnostics.parquet"
        )
        diag_path = Path(output_dir) / diag_name
        save_parquet(diagnostics, diag_path)
        self.write_forceplate_fz_qc_summary_csv(diag_path=diag_path, output_dir=output_dir)

        ch_to_unloaded = {
            "Fx": "Fx_unloaded_100hz",
            "Fy": "Fy_unloaded_100hz",
            "Fz": "Fz_unloaded_100hz",
            "Mx": "Mx_unloaded_100hz",
            "My": "My_unloaded_100hz",
            "Mz": "Mz_unloaded_100hz",
        }
        unloaded_cols = [ch_to_unloaded[c] for c in apply_channels]

        diag_lf = diagnostics.select(
            keys
            + [
                "mocap_idx_local",
                "applied",
                *unloaded_cols,
            ]
        ).lazy()

        with_sub = (
            dataset_lf.with_columns(
                (pl.col("DeviceFrame") // pl.lit(frame_ratio)).cast(pl.Int64).alias("mocap_idx_local")
            )
            .join(diag_lf, on=keys + ["mocap_idx_local"], how="left")
        )

        updates: list[pl.Expr] = []
        for ch in apply_channels:
            unloaded_col = ch_to_unloaded[ch]
            updates.append(
                (
                    pl.col(ch).cast(pl.Float64, strict=False)
                    - pl.col(unloaded_col)
                    .cast(pl.Float64, strict=False)
                    .fill_nan(0.0)
                    .fill_null(0.0)
                ).alias(ch)
            )
        if updates:
            with_sub = with_sub.with_columns(updates)

        cop_thr = float((fp_cfg.get("cop", {}) or {}).get("fz_threshold_n", 20.0))
        ok_fz = pl.col("Fz").cast(pl.Float64, strict=False).abs() >= pl.lit(float(cop_thr))
        with_sub = with_sub.with_columns(
            [
                pl.when(ok_fz)
                .then((-pl.col("My").cast(pl.Float64, strict=False)) / pl.col("Fz").cast(pl.Float64, strict=False))
                .otherwise(None)
                .alias("Cx"),
                pl.when(ok_fz)
                .then((pl.col("Mx").cast(pl.Float64, strict=False)) / pl.col("Fz").cast(pl.Float64, strict=False))
                .otherwise(None)
                .alias("Cy"),
                pl.lit(0.0).cast(pl.Float64).alias("Cz"),
            ]
        )

        drop_cols = [
            c
            for c in [
                "mocap_idx_local",
                "applied",
                *unloaded_cols,
            ]
            if c in with_sub.collect_schema().names()
        ]
        if drop_cols:
            with_sub = with_sub.drop(drop_cols)

        logger.info(f"Direct subtract 완료: Fx/My/Mz 및 Cx/Cy/Cz 덮어쓰기, 진단 파일 저장: {diag_path}")
        return with_sub

    def _try_build_unloaded_velocity_templates_subtract(self) -> dict[int, _SubtractTemplate100Hz]:
        cached = getattr(self, "_subtract_templates_cache", None)
        if isinstance(cached, dict):
            return cached

        fp_cfg = self.config.get("forceplate", {}) or {}
        sub_cfg = fp_cfg.get("subtract", {}) or {}

        unload_data_dir = Path(str(sub_cfg.get("unload_data_dir", "Archive/unload_data")))
        if not unload_data_dir.is_absolute():
            unload_data_dir = (MODULE_DIR / unload_data_dir).resolve()
        if not unload_data_dir.exists():
            logger.warning(f"Direct subtract template skip: unload_data_dir not found: {unload_data_dir}")
            self._subtract_templates_cache = {}
            return {}

        timing_xlsx_name = str(sub_cfg.get("timing_xlsx_name", "FP_platform_on-offset.xlsx"))
        timing_sheet = str(sub_cfg.get("timing_sheet", "Sheet1"))
        timing_path = (unload_data_dir / timing_xlsx_name).resolve()
        if not timing_path.exists():
            logger.warning(f"Direct subtract template skip: timing xlsx not found: {timing_path}")
            self._subtract_templates_cache = {}
            return {}

        timing_cols = sub_cfg.get("timing_columns", {}) or {}
        col_velocity = str(timing_cols.get("velocity", "velocity"))
        col_trial = str(timing_cols.get("trial", "trial"))
        col_onset = str(timing_cols.get("onset", "onset"))
        col_offset = str(timing_cols.get("offset", "offset"))

        timing = pl.read_excel(timing_path, sheet_name=timing_sheet, engine="openpyxl")
        timing = timing.rename({c: str(c).strip() for c in timing.columns})
        # Subtract는 velocity별 unload_range_frames(=offset-onset)를 필요로 하므로
        # timing xlsx에 onset/offset이 반드시 존재해야 한다.
        need_cols = {col_velocity, col_trial, col_onset, col_offset}
        missing = sorted(need_cols - set(timing.columns))
        if missing:
            raise ValueError(f"timing xlsx에 필요한 컬럼이 없습니다: {missing} (path={timing_path})")

        base = timing.select(
            [
                pl.col(col_velocity).alias("velocity"),
                pl.col(col_trial).alias("trial"),
                pl.col(col_onset).alias("onset"),
                pl.col(col_offset).alias("offset"),
            ]
        ).drop_nulls(["velocity", "trial", "onset", "offset"])

        base = base.with_columns(
            [
                pl.col("velocity").cast(pl.Float64, strict=False).alias("velocity"),
                pl.col("trial").cast(pl.Int64, strict=False).alias("trial"),
                pl.col("onset").cast(pl.Int64, strict=False).alias("onset"),
                pl.col("offset").cast(pl.Int64, strict=False).alias("offset"),
                pl.col("velocity").cast(pl.Float64, strict=False).round(0).cast(pl.Int64).alias("velocity_int"),
            ]
        )
        if base.is_empty():
            logger.warning(f"Direct subtract template skip: timing table is empty after cleanup: {timing_path}")
            self._subtract_templates_cache = {}
            return {}

        # Archive/test/validation 등의 외부 산출물에 의존하지 않고,
        # timing xlsx의 offset-onset으로 velocity별 unload_range_frames(기본: median)를 계산한다.
        range_map: dict[int, int] = {}
        ranges = (
            base.with_columns((pl.col("offset") - pl.col("onset")).cast(pl.Int64).alias("unload_range_frames"))
            .filter(pl.col("unload_range_frames").is_not_null())
            .group_by("velocity_int")
            .agg(pl.col("unload_range_frames").median().round(0).cast(pl.Int64).alias("unload_range_frames"))
        )
        for r in ranges.iter_rows(named=True):
            v_int = int(r["velocity_int"])
            unload_range_frames = int(r["unload_range_frames"])
            if unload_range_frames < 0:
                continue
            range_map[v_int] = unload_range_frames

        if not range_map:
            logger.warning("Direct subtract template skip: unload_range_frames map is empty")
            self._subtract_templates_cache = {}
            return {}

        pattern = str(sub_cfg.get("raw_forceplate_filename_pattern", "*_perturb_{velocity}_{trial:03d}_forceplate_3.csv"))
        parse_cfg = sub_cfg.get("raw_forceplate_parse", {}) or {}
        header_startswith = str(parse_cfg.get("header_startswith", "MocapFrame"))
        required_columns = list(parse_cfg.get("required_columns") or [])
        if not required_columns:
            required_columns = ["MocapFrame", "DeviceFrame", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]

        aggregate = str(sub_cfg.get("aggregate", "mean")).strip().lower()
        if aggregate not in {"mean", "median"}:
            raise ValueError(f"forceplate.subtract.aggregate={aggregate!r} (지원: 'mean'|'median')")

        encodings = list((self.config.get("file_io", {}) or {}).get("encodings") or ["utf-8-sig", "utf-8"])
        frame_ratio = get_frame_ratio(self.config)

        templates: dict[int, _SubtractTemplate100Hz] = {}
        velocity_ints = (
            base.select(pl.col("velocity_int").drop_nulls().unique().sort())
            .get_column("velocity_int")
            .to_list()
        )
        for v_int in velocity_ints:
            v_int = int(v_int)
            unload_range_frames = range_map.get(v_int)
            if unload_range_frames is None:
                continue

            rows = base.filter(pl.col("velocity_int") == v_int).select(["trial", "onset"]).to_dicts()
            pieces: list[pl.DataFrame] = []
            used_files: list[str] = []

            for rr in rows:
                trial = int(rr["trial"])
                onset = int(rr["onset"])
                glob_pat = pattern.format(velocity=v_int, trial=trial)
                matches = sorted(unload_data_dir.glob(glob_pat))
                if not matches:
                    continue
                csv_path = matches[0]

                try:
                    raw_df = self._read_forceplate_csv_table(
                        csv_path,
                        header_startswith=header_startswith,
                        required_columns=required_columns,
                        encodings=encodings,
                    )
                except Exception as e:
                    logger.warning(f"skip unload csv (read failed): {csv_path} ({e})")
                    continue

                if raw_df.is_empty():
                    continue

                df100 = (
                    raw_df.with_columns(
                        [
                            pl.col("DeviceFrame").cast(pl.Int64, strict=False).alias("DeviceFrame"),
                            pl.col("MocapFrame").cast(pl.Int64, strict=False).alias("MocapFrame"),
                            (pl.col("DeviceFrame") // pl.lit(frame_ratio)).cast(pl.Int64).alias("mocap_idx_local"),
                        ]
                    )
                    .group_by("mocap_idx_local")
                    .agg(
                        [
                            pl.col("MocapFrame").first().alias("MocapFrame"),
                            pl.col("Fx").cast(pl.Float64, strict=False).mean().alias("Fx"),
                            pl.col("Fy").cast(pl.Float64, strict=False).mean().alias("Fy"),
                            pl.col("Fz").cast(pl.Float64, strict=False).mean().alias("Fz"),
                            pl.col("Mx").cast(pl.Float64, strict=False).mean().alias("Mx"),
                            pl.col("My").cast(pl.Float64, strict=False).mean().alias("My"),
                            pl.col("Mz").cast(pl.Float64, strict=False).mean().alias("Mz"),
                        ]
                    )
                    .sort("mocap_idx_local")
                )

                mocap_frame = df100.get_column("MocapFrame").to_numpy()
                fx = df100.get_column("Fx").to_numpy()
                fy = df100.get_column("Fy").to_numpy()
                fz = df100.get_column("Fz").to_numpy()
                mx = df100.get_column("Mx").to_numpy()
                my = df100.get_column("My").to_numpy()
                mz = df100.get_column("Mz").to_numpy()

                onset_local = _nearest_index(mocap_frame, onset)
                offset_local = _nearest_index(mocap_frame, onset + int(unload_range_frames))
                if offset_local < onset_local:
                    continue

                rel = (mocap_frame[onset_local : offset_local + 1] - mocap_frame[onset_local]).astype(int)
                piece = pl.DataFrame(
                    {
                        "relative_frame": rel,
                        "Fx": fx[onset_local : offset_local + 1].astype(float),
                        "Fy": fy[onset_local : offset_local + 1].astype(float),
                        "Fz": fz[onset_local : offset_local + 1].astype(float),
                        "Mx": mx[onset_local : offset_local + 1].astype(float),
                        "My": my[onset_local : offset_local + 1].astype(float),
                        "Mz": mz[onset_local : offset_local + 1].astype(float),
                    }
                )
                pieces.append(piece)
                used_files.append(csv_path.name)

            if not pieces:
                continue

            stacked = pl.concat(pieces, how="vertical_relaxed")
            if aggregate == "median":
                agg_df = (
                    stacked.group_by("relative_frame")
                    .agg(
                        [
                            pl.col("Fx").median().alias("Fx"),
                            pl.col("Fy").median().alias("Fy"),
                            pl.col("Fz").median().alias("Fz"),
                            pl.col("Mx").median().alias("Mx"),
                            pl.col("My").median().alias("My"),
                            pl.col("Mz").median().alias("Mz"),
                        ]
                    )
                    .sort("relative_frame")
                )
            else:
                agg_df = (
                    stacked.group_by("relative_frame")
                    .agg(
                        [
                            pl.col("Fx").mean().alias("Fx"),
                            pl.col("Fy").mean().alias("Fy"),
                            pl.col("Fz").mean().alias("Fz"),
                            pl.col("Mx").mean().alias("Mx"),
                            pl.col("My").mean().alias("My"),
                            pl.col("Mz").mean().alias("Mz"),
                        ]
                    )
                    .sort("relative_frame")
                )

            full = pl.DataFrame({"relative_frame": np.arange(int(unload_range_frames) + 1, dtype=np.int64)})
            agg_df = full.join(agg_df, on="relative_frame", how="left").sort("relative_frame")
            agg_df = agg_df.with_columns(
                [
                    pl.col("Fx").interpolate().fill_null(strategy="forward").fill_null(strategy="backward"),
                    pl.col("Fy").interpolate().fill_null(strategy="forward").fill_null(strategy="backward"),
                    pl.col("Fz").interpolate().fill_null(strategy="forward").fill_null(strategy="backward"),
                    pl.col("Mx").interpolate().fill_null(strategy="forward").fill_null(strategy="backward"),
                    pl.col("My").interpolate().fill_null(strategy="forward").fill_null(strategy="backward"),
                    pl.col("Mz").interpolate().fill_null(strategy="forward").fill_null(strategy="backward"),
                ]
            )

            fx_arr = agg_df.get_column("Fx").to_numpy()
            fy_arr = agg_df.get_column("Fy").to_numpy()
            fz_arr = agg_df.get_column("Fz").to_numpy()
            mx_arr = agg_df.get_column("Mx").to_numpy()
            my_arr = agg_df.get_column("My").to_numpy()
            mz_arr = agg_df.get_column("Mz").to_numpy()

            if fx_arr.size:
                fx_arr = fx_arr - float(fx_arr[0])
            if fy_arr.size:
                fy_arr = fy_arr - float(fy_arr[0])
            if fz_arr.size:
                fz_arr = fz_arr - float(fz_arr[0])
            if mx_arr.size:
                mx_arr = mx_arr - float(mx_arr[0])
            if my_arr.size:
                my_arr = my_arr - float(my_arr[0])
            if mz_arr.size:
                mz_arr = mz_arr - float(mz_arr[0])

            templates[v_int] = _SubtractTemplate100Hz(
                velocity_int=int(v_int),
                unload_range_frames=int(unload_range_frames),
                fx=fx_arr,
                fy=fy_arr,
                fz=fz_arr,
                mx=mx_arr,
                my=my_arr,
                mz=mz_arr,
                n_trials=len(pieces),
                meta={
                    "unload_data_dir": str(unload_data_dir),
                    "timing_xlsx": str(timing_path),
                    "pattern": pattern,
                    "files": used_files,
                    "aggregate": aggregate,
                },
            )

        self._subtract_templates_cache = templates
        return templates

    def _read_forceplate_csv_table(
        self,
        path: Path,
        *,
        header_startswith: str,
        required_columns: list[str],
        encodings: list[str],
    ) -> pl.DataFrame:
        import io

        path = Path(path)
        header_row: int | None = None
        used_enc: str | None = None

        for enc in encodings:
            try:
                with path.open("r", encoding=enc) as f:
                    for i, line in enumerate(f):
                        if str(line).strip().startswith(str(header_startswith)):
                            header_row = int(i)
                            used_enc = str(enc)
                            break
                if header_row is not None:
                    break
            except UnicodeError:
                continue

        if header_row is None or used_enc is None:
            raise ValueError(f"csv header not found: {path} (startswith={header_startswith!r})")

        try:
            df = pl.read_csv(path, skip_rows=int(header_row), encoding=str(used_enc), ignore_errors=True)
        except TypeError:
            with path.open("r", encoding=str(used_enc), errors="replace") as f:
                for _ in range(int(header_row)):
                    next(f, None)
                df = pl.read_csv(io.StringIO(f.read()), ignore_errors=True)

        df = df.rename({c: str(c).strip() for c in df.columns})
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"missing columns in {path.name}: {missing}")
        df = df.select(required_columns)
        return self._apply_forceplate_axis_transform(df)

    def _compute_subtract_diagnostics_from_trial_lists(
        self,
        trials_df: pl.DataFrame,
        *,
        templates: dict[int, _SubtractTemplate100Hz],
    ) -> pl.DataFrame:
        fp_cfg = self.config.get("forceplate", {}) or {}
        sub_cfg = fp_cfg.get("subtract", {}) or {}
        missing_policy = str(sub_cfg.get("missing_velocity_policy", "skip")).strip().lower()
        if missing_policy not in {"skip", "nearest", "interpolate"}:
            raise ValueError(
                f"forceplate.subtract.missing_velocity_policy={missing_policy!r} (지원: 'skip'|'nearest'|'interpolate')"
            )

        template_keys = sorted(int(k) for k in templates.keys())

        parts: list[pl.DataFrame] = []
        for row in trials_df.iter_rows(named=True):
            subject = str(row["subject"])
            velocity = float(row["velocity"])
            trial_num = int(row["trial_num"])

            mocap_idx = np.asarray(row["mocap_idx_local"], dtype=int)
            mocap_frame = np.asarray(row["MocapFrame"], dtype=int)

            fx_meas = np.asarray(row["Fx"], dtype=float)
            fy_meas = np.asarray(row["Fy"], dtype=float) if "Fy" in row else np.full_like(fx_meas, np.nan, dtype=float)
            my_meas = np.asarray(row["My"], dtype=float)
            mz_meas = np.asarray(row["Mz"], dtype=float)
            mx_meas = np.asarray(row["Mx"], dtype=float) if "Mx" in row else np.full_like(fx_meas, np.nan, dtype=float)
            fz_meas = np.asarray(row["Fz"], dtype=float) if "Fz" in row else np.full_like(fx_meas, np.nan, dtype=float)

            onset = int(row.get("platform_onset")) if row.get("platform_onset") is not None else int(mocap_frame[0])
            offset = int(row.get("platform_offset")) if row.get("platform_offset") is not None else int(mocap_frame[-1])
            onset_local = _nearest_index(mocap_frame, onset)
            offset_local = _nearest_index(mocap_frame, offset)
            if offset_local < onset_local:
                onset_local, offset_local = offset_local, onset_local

            n100 = int(fx_meas.size)
            human_duration = int(offset_local - onset_local + 1)
            if human_duration < 0:
                human_duration = 0
            unload_range_frames = int(max(0, human_duration - 1))

            v_int = int(round(velocity))
            tmpl = templates.get(v_int)
            template_policy = "exact"
            template_velocity_int_used: int | None = int(v_int) if tmpl is not None else None
            template_velocity_int_lo: int | None = None
            template_velocity_int_hi: int | None = None
            template_interp_weight: float | None = None

            if tmpl is None and missing_policy != "skip" and template_keys:
                v_val = float(velocity)
                if missing_policy == "nearest":
                    k_used = min(template_keys, key=lambda k: (abs(float(k) - v_val), k))
                    tmpl = templates.get(int(k_used))
                    template_policy = "nearest"
                    template_velocity_int_used = int(k_used)
                else:
                    lower = [k for k in template_keys if float(k) <= v_val]
                    upper = [k for k in template_keys if float(k) >= v_val]
                    if not lower or not upper:
                        k_used = min(template_keys, key=lambda k: (abs(float(k) - v_val), k))
                        tmpl = templates.get(int(k_used))
                        template_policy = "nearest"
                        template_velocity_int_used = int(k_used)
                    else:
                        k_lo = int(lower[-1])
                        k_hi = int(upper[0])
                        if k_lo == k_hi:
                            tmpl = templates.get(int(k_lo))
                            template_policy = "nearest"
                            template_velocity_int_used = int(k_lo)
                        else:
                            tmpl = None
                            template_policy = "interpolate"
                            template_velocity_int_used = None
                            template_velocity_int_lo = int(k_lo)
                            template_velocity_int_hi = int(k_hi)
                            template_interp_weight = float((v_val - float(k_lo)) / float(k_hi - k_lo))

            template_available = bool((tmpl is not None) or (template_policy == "interpolate"))
            applied = False
            missing_reason = None
            lag_frames = 0
            corr = float("nan")
            if tmpl is not None:
                tmpl_n_trials = int(tmpl.n_trials)
            elif template_policy == "interpolate" and template_velocity_int_lo is not None and template_velocity_int_hi is not None:
                tmpl_n_trials = int(
                    int(templates[int(template_velocity_int_lo)].n_trials) + int(templates[int(template_velocity_int_hi)].n_trials)
                )
            else:
                tmpl_n_trials = 0

            fx_unloaded = np.full(n100, np.nan, dtype=float)
            fy_unloaded = np.full(n100, np.nan, dtype=float)
            fz_unloaded = np.full(n100, np.nan, dtype=float)
            mx_unloaded = np.full(n100, np.nan, dtype=float)
            my_unloaded = np.full(n100, np.nan, dtype=float)
            mz_unloaded = np.full(n100, np.nan, dtype=float)

            if tmpl is None and template_policy != "interpolate":
                missing_reason = "no_unloaded_template_for_velocity"
                template_policy = "skip"
                fx_corr = fx_meas.astype(float, copy=True)
                fy_corr = fy_meas.astype(float, copy=True)
                fz_corr = fz_meas.astype(float, copy=True)
                mx_corr = mx_meas.astype(float, copy=True)
                my_corr = my_meas.astype(float, copy=True)
                mz_corr = mz_meas.astype(float, copy=True)
            else:
                if tmpl is not None:
                    fx_seg = np.asarray(tmpl.fx, dtype=float)
                    fy_seg = np.asarray(tmpl.fy, dtype=float)
                    fz_seg = np.asarray(tmpl.fz, dtype=float)
                    mx_seg = np.asarray(tmpl.mx, dtype=float)
                    my_seg = np.asarray(tmpl.my, dtype=float)
                    mz_seg = np.asarray(tmpl.mz, dtype=float)
                else:
                    tmpl_lo = templates[int(template_velocity_int_lo)]
                    tmpl_hi = templates[int(template_velocity_int_hi)]
                    w = float(template_interp_weight or 0.0)

                    fx_lo = np.asarray(tmpl_lo.fx, dtype=float)
                    fy_lo = np.asarray(tmpl_lo.fy, dtype=float)
                    fz_lo = np.asarray(tmpl_lo.fz, dtype=float)
                    mx_lo = np.asarray(tmpl_lo.mx, dtype=float)
                    my_lo = np.asarray(tmpl_lo.my, dtype=float)
                    mz_lo = np.asarray(tmpl_lo.mz, dtype=float)

                    fx_hi = np.asarray(tmpl_hi.fx, dtype=float)
                    fy_hi = np.asarray(tmpl_hi.fy, dtype=float)
                    fz_hi = np.asarray(tmpl_hi.fz, dtype=float)
                    mx_hi = np.asarray(tmpl_hi.mx, dtype=float)
                    my_hi = np.asarray(tmpl_hi.my, dtype=float)
                    mz_hi = np.asarray(tmpl_hi.mz, dtype=float)

                    target_len = int(
                        max(
                            fx_lo.size,
                            fy_lo.size,
                            fz_lo.size,
                            mx_lo.size,
                            my_lo.size,
                            mz_lo.size,
                            fx_hi.size,
                            fy_hi.size,
                            fz_hi.size,
                            mx_hi.size,
                            my_hi.size,
                            mz_hi.size,
                        )
                    )
                    if fx_lo.size < target_len:
                        fx_lo = np.pad(fx_lo, (0, target_len - fx_lo.size), mode="edge")
                    if fy_lo.size < target_len:
                        fy_lo = np.pad(fy_lo, (0, target_len - fy_lo.size), mode="edge")
                    if fz_lo.size < target_len:
                        fz_lo = np.pad(fz_lo, (0, target_len - fz_lo.size), mode="edge")
                    if mx_lo.size < target_len:
                        mx_lo = np.pad(mx_lo, (0, target_len - mx_lo.size), mode="edge")
                    if my_lo.size < target_len:
                        my_lo = np.pad(my_lo, (0, target_len - my_lo.size), mode="edge")
                    if mz_lo.size < target_len:
                        mz_lo = np.pad(mz_lo, (0, target_len - mz_lo.size), mode="edge")
                    if fx_hi.size < target_len:
                        fx_hi = np.pad(fx_hi, (0, target_len - fx_hi.size), mode="edge")
                    if fy_hi.size < target_len:
                        fy_hi = np.pad(fy_hi, (0, target_len - fy_hi.size), mode="edge")
                    if fz_hi.size < target_len:
                        fz_hi = np.pad(fz_hi, (0, target_len - fz_hi.size), mode="edge")
                    if mx_hi.size < target_len:
                        mx_hi = np.pad(mx_hi, (0, target_len - mx_hi.size), mode="edge")
                    if my_hi.size < target_len:
                        my_hi = np.pad(my_hi, (0, target_len - my_hi.size), mode="edge")
                    if mz_hi.size < target_len:
                        mz_hi = np.pad(mz_hi, (0, target_len - mz_hi.size), mode="edge")

                    fx_seg = (1.0 - w) * fx_lo + w * fx_hi
                    fy_seg = (1.0 - w) * fy_lo + w * fy_hi
                    fz_seg = (1.0 - w) * fz_lo + w * fz_hi
                    mx_seg = (1.0 - w) * mx_lo + w * mx_hi
                    my_seg = (1.0 - w) * my_lo + w * my_hi
                    mz_seg = (1.0 - w) * mz_lo + w * mz_hi

                template_len = int(fx_seg.size)
                unload_range_frames = int(max(0, template_len - 1))

                if human_duration > 0 and template_len > 0:
                    head_len = int(min(human_duration, template_len))
                    end_idx = onset_local + head_len
                    fx_unloaded[onset_local:end_idx] = fx_seg[:head_len]
                    fy_unloaded[onset_local:end_idx] = fy_seg[:head_len]
                    fz_unloaded[onset_local:end_idx] = fz_seg[:head_len]
                    mx_unloaded[onset_local:end_idx] = mx_seg[:head_len]
                    my_unloaded[onset_local:end_idx] = my_seg[:head_len]
                    mz_unloaded[onset_local:end_idx] = mz_seg[:head_len]

                    if human_duration > template_len:
                        tail_start = onset_local + template_len
                        tail_end = offset_local + 1
                        if tail_start < tail_end:
                            last_fx = float(fx_seg[template_len - 1])
                            last_fy = float(fy_seg[template_len - 1])
                            last_fz = float(fz_seg[template_len - 1])
                            last_mx = float(mx_seg[template_len - 1])
                            last_my = float(my_seg[template_len - 1])
                            last_mz = float(mz_seg[template_len - 1])
                            fx_unloaded[tail_start:tail_end] = last_fx
                            fy_unloaded[tail_start:tail_end] = last_fy
                            fz_unloaded[tail_start:tail_end] = last_fz
                            mx_unloaded[tail_start:tail_end] = last_mx
                            my_unloaded[tail_start:tail_end] = last_my
                            mz_unloaded[tail_start:tail_end] = last_mz

                    if offset_local + 1 < n100:
                        last_fx = float(fx_seg[template_len - 1])
                        last_fy = float(fy_seg[template_len - 1])
                        last_fz = float(fz_seg[template_len - 1])
                        last_mx = float(mx_seg[template_len - 1])
                        last_my = float(my_seg[template_len - 1])
                        last_mz = float(mz_seg[template_len - 1])
                        fx_unloaded[offset_local + 1 :] = last_fx
                        fy_unloaded[offset_local + 1 :] = last_fy
                        fz_unloaded[offset_local + 1 :] = last_fz
                        mx_unloaded[offset_local + 1 :] = last_mx
                        my_unloaded[offset_local + 1 :] = last_my
                        mz_unloaded[offset_local + 1 :] = last_mz

                fx_corr = fx_meas - np.nan_to_num(fx_unloaded, nan=0.0)
                fy_corr = fy_meas - np.nan_to_num(fy_unloaded, nan=0.0)
                fz_corr = fz_meas - np.nan_to_num(fz_unloaded, nan=0.0)
                mx_corr = mx_meas - np.nan_to_num(mx_unloaded, nan=0.0)
                my_corr = my_meas - np.nan_to_num(my_unloaded, nan=0.0)
                mz_corr = mz_meas - np.nan_to_num(mz_unloaded, nan=0.0)

                applied = True

            out = pl.DataFrame(
                {
                    "subject": [subject] * n100,
                    "velocity": [velocity] * n100,
                    "trial_num": [trial_num] * n100,
                    "mocap_idx_local": mocap_idx.astype(int),
                    "MocapFrame": mocap_frame.astype(int),
                    "velocity_int": [v_int] * n100,
                    "unload_range_frames": [unload_range_frames] * n100,
                    "template_available": [bool(template_available)] * n100,
                    "applied": [bool(applied)] * n100,
                    "template_policy": [str(template_policy)] * n100,
                    "template_velocity_int_used": [template_velocity_int_used] * n100,
                    "template_velocity_int_lo": [template_velocity_int_lo] * n100,
                    "template_velocity_int_hi": [template_velocity_int_hi] * n100,
                    "template_interp_weight": [template_interp_weight] * n100,
                    "missing_reason": [missing_reason] * n100,
                    "template_n_trials": [int(tmpl_n_trials)] * n100,
                    "Fx_measured_100hz": fx_meas.astype(float),
                    "Fy_measured_100hz": fy_meas.astype(float),
                    "Fz_measured_100hz": fz_meas.astype(float),
                    "Mx_measured_100hz": mx_meas.astype(float),
                    "My_measured_100hz": my_meas.astype(float),
                    "Mz_measured_100hz": mz_meas.astype(float),
                    "Fx_unloaded_100hz": fx_unloaded.astype(float),
                    "Fy_unloaded_100hz": fy_unloaded.astype(float),
                    "Fz_unloaded_100hz": fz_unloaded.astype(float),
                    "Mx_unloaded_100hz": mx_unloaded.astype(float),
                    "My_unloaded_100hz": my_unloaded.astype(float),
                    "Mz_unloaded_100hz": mz_unloaded.astype(float),
                    "Fx_corrected_100hz": fx_corr.astype(float),
                    "Fy_corrected_100hz": fy_corr.astype(float),
                    "Fz_corrected_100hz": fz_corr.astype(float),
                    "Mx_corrected_100hz": mx_corr.astype(float),
                    "My_corrected_100hz": my_corr.astype(float),
                    "Mz_corrected_100hz": mz_corr.astype(float),
                    "unload_template_lag_frames": [int(lag_frames)] * n100,
                    "unload_template_corr": [float(corr)] * n100,
                    "onset_local_100hz": [int(onset_local)] * n100,
                    "offset_local_100hz": [int(offset_local)] * n100,
                }
            )
            parts.append(out)

        if not parts:
            return pl.DataFrame()
        return pl.concat(parts, how="vertical_relaxed")

    def save_dataset(self, df: pl.DataFrame, output_dir: Path) -> None:
        """
        Save the comprehensive dataset to Parquet.

        Args:
            df: Dataset DataFrame
            output_dir: Output directory path
        """
        logger.info("Saving comprehensive dataset...")

        # Create output directory
        create_output_directory(output_dir)

        # Reorder/drop columns to match legacy canonical output
        df_cols = df.collect_schema().names() if isinstance(df, pl.LazyFrame) else list(df.columns)
        emg_cols = [c for c in self.muscle_names if c in df_cols]
        fp_merge_cols = self.config.get("forceplate", {}).get("merge_columns", [])
        fp_cols = [c for c in fp_merge_cols if c in df_cols]
        fp_zero_cfg = self.config.get("forceplate", {}).get("zeroed", {}) or {}
        fp_zero_enabled = bool(fp_zero_cfg.get("enabled", False))
        fp_zero_cols: list[str] = []
        if fp_zero_enabled:
            cols_cfg = fp_zero_cfg.get("columns")
            if cols_cfg is None:
                base_cols = list(fp_merge_cols)
            elif isinstance(cols_cfg, list):
                base_cols = [str(c).strip() for c in cols_cfg]
            else:
                raise ValueError("forceplate.zeroed.columns는 list[str] 이어야 합니다.")
            suffix = str(fp_zero_cfg.get("suffix", "_zero"))
            fp_zero_cols = [f"{c}{suffix}" for c in base_cols if f"{c}{suffix}" in df_cols]
        canonical_order = (
            ["MocapFrame", "MocapTime", "DeviceFrame"]
            + emg_cols
            + fp_cols
            + fp_zero_cols
            + [
                "date",
                "subject",
                "velocity",
                "trial_num",
                "task",
                "intervention",
                "original_DeviceFrame",
                "state",
                "step_TF",
                "RPS",
                "mixed",
                "platform_onset",
                "platform_offset",
            ]
        )
        canonical_present = [c for c in canonical_order if c in df_cols]
        canonical_set = set(canonical_present)
        select_cols = canonical_present + [c for c in df_cols if c not in canonical_set]
        df = df.select(select_cols)

        parquet_name = (
            self.config.get("pipeline_files", {}).get("stage01_merged_dataset")
            or "merged_data_comprehensive.parquet"
        )
        parquet_path = output_dir / parquet_name

        # Support streaming output (LazyFrame) to avoid OOM
        save_parquet(df, parquet_path)

        if isinstance(df, pl.LazyFrame):
            n_rows = int(df.select(pl.len()).collect(engine="streaming").to_series()[0])
            cols = df.collect_schema().names()
        else:
            n_rows = int(df.height)
            cols = list(df.columns)

        logger.info(f"Saved comprehensive Parquet: {parquet_path} ({n_rows} rows)")

        # Save summary
        summary_path = output_dir / "dataset_build_summary.txt"
        with open(summary_path, 'w', encoding='utf-8-sig') as f:
            f.write("="*60 + "\n")
            f.write("Stage 01: Build Comprehensive Dataset Summary\n")
            f.write("="*60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Pre-onset frames: {self.pre_frames}\n")
            f.write(f"  Post-offset frames: {self.post_frames}\n")
            f.write(f"  Frame ratio: {get_frame_ratio(self.config)}:1 (DeviceFrame:MocapFrame)\n\n")
            f.write(f"Statistics:\n")
            f.write(f"  Total trials: {self.stats['total_trials']}\n")
            f.write(f"  Segments created: {self.stats['segments_created']}\n")
            f.write(f"  Total rows: {n_rows}\n")
            f.write(f"  Total columns: {len(cols)}\n")
            f.write(f"  Errors: {len(self.stats['errors'])}\n\n")
            missing_n = int(self.stats.get("missing_units_n", 0) or 0)
            f.write(f"  Missing units: {missing_n}\n\n")
            f.write(f"Output files:\n")
            f.write(f"  Parquet: {parquet_path}\n\n")
            f.write(f"Columns:\n")
            for col in cols:
                f.write(f"  - {col}\n")

            missing_detail = self.stats.get("missing_units_detail_df")
            if isinstance(missing_detail, pl.DataFrame) and missing_detail.height > 0:
                f.write("\nMissing units detail (platform sheet 기준):\n")
                for rr in missing_detail.iter_rows(named=True):
                    f.write(
                        "  - "
                        f"subject={rr['subject']}, velocity={rr['velocity']}, trial_num={rr['trial_num']}, "
                        f"reason={rr['reason']}, "
                        f"start_req={rr.get('start_frame_req')}, end_req={rr.get('end_frame_req')}, "
                        f"raw_min={rr.get('min_raw')}, raw_max={rr.get('max_raw')}\n"
                    )

            if self.stats['errors']:
                f.write("\nErrors encountered:\n")
                for error in self.stats['errors'][:10]:  # Show first 10 errors
                    f.write(f"  - {error}\n")

        logger.info(f"Saved summary: {summary_path}")

    def run(self, output_dir: Path) -> bool:
        """
        Run the complete dataset building process.

        Args:
            output_dir: Output directory path

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("="*60)
            logger.info("Starting Stage 01: Build Comprehensive Dataset")
            logger.info("="*60)

            # Build dataset (lazy, streamable)
            dataset_lf = self.build_dataset_lazy()

            # Merge platform metadata (lazy)
            dataset_lf = self.merge_platform_metadata_lazy(dataset_lf)

            # Ensure Stage1(window correction) correction table exists (no external dependency)
            self.ensure_forceplate_stage1_corrections(output_dir=output_dir)

            # Apply Stage1(window correction) shift to forceplate signals (lazy)
            dataset_lf = self.apply_forceplate_stage1_corrections_lazy(dataset_lf, output_dir=output_dir)

            # Apply forceplate inertial removal (direct subtract) + COP overwrite (lazy join/broadcast)
            dataset_lf = self.apply_forceplate_inertial_removal_lazy(dataset_lf, output_dir)

            # Create forceplate *_zero columns (pre-onset baseline median)
            dataset_lf = self.apply_forceplate_zeroed_columns_lazy(dataset_lf)

            # Save dataset (streaming Parquet sink)
            self.save_dataset(dataset_lf, output_dir)

            logger.info("="*60)
            logger.info("✅ Stage 01: Build Comprehensive Dataset - COMPLETED")
            logger.info("="*60)

            return True

        except Exception as e:
            logger.error(f"Stage 01 failed: {e}")
            logger.debug(traceback.format_exc())
            return False
