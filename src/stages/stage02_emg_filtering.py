#!/usr/bin/env python3
"""
Stage 02: EMG Filtering

Performs EMG signal processing (HPF → demean → rectify → LPF).
Processing options are controlled by `config.yaml`.
"""

import argparse
import logging
import multiprocessing
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt

from src.config_helpers import get_device_hz, get_frame_ratio, get_output_path, load_config_yaml
from src.utils import get_logger, log_and_print, read_parquet_robust, save_parquet

logger = get_logger("02_emg_filtering")

import warnings

warnings.filterwarnings("ignore")

# Prevent BLAS oversubscription during parallel filtering
for _var in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(_var, "1")


# ==================== COM MERGE HELPERS (Stage02-local) ====================


@dataclass(frozen=True)
class ComTrialMeta:
    subject: str
    date: str | None
    velocity: float
    trial_num: int
    start_mocap: int
    end_mocap: int


def _velocity_key(value: float) -> float:
    try:
        return round(float(value), 6)
    except Exception:
        return float("nan")


def _log(logger: logging.Logger | None, level: int, message: str) -> None:
    if logger is None:
        return
    logger.log(level, message)


def _index_com_files(
    raw_root: Path, file_regex: str, logger: logging.Logger | None
) -> dict[tuple[str, str, float, int], Path]:
    raw_root = Path(raw_root)
    if not raw_root.exists():
        _log(logger, logging.WARNING, f"COM raw_data_root not found: {raw_root}")
        return {}

    pattern = re.compile(str(file_regex))
    out: dict[tuple[str, str, float, int], Path] = {}

    for path in raw_root.rglob("*"):
        if not path.is_file():
            continue
        m = pattern.match(path.name)
        if not m:
            continue

        subject = path.parent.name
        date = str(m.groupdict().get("date") or "").strip()
        velocity_raw = m.groupdict().get("velocity")
        trial_raw = m.groupdict().get("trial")
        if not date or velocity_raw is None or trial_raw is None:
            continue

        try:
            velocity = _velocity_key(float(str(velocity_raw).strip()))
            trial_num = int(str(trial_raw).strip())
        except Exception:
            continue

        key = (subject, date, velocity, trial_num)
        if key in out and out[key] != path:
            _log(logger, logging.WARNING, f"Duplicate COM file for {key}: keeping {out[key].name}, skipping {path.name}")
            continue
        out[key] = path

    return out


def _read_com_excel(path: Path, com_cfg: dict) -> pl.DataFrame:
    import pandas as pd

    excel_cfg = (com_cfg.get("excel", {}) or {}) if isinstance(com_cfg, dict) else {}
    skiprows = int(excel_cfg.get("skiprows", 1))
    sheet_name: Any = excel_cfg.get("sheet_name", 0)

    cols_cfg = (com_cfg.get("columns", {}) or {}) if isinstance(com_cfg, dict) else {}
    frame_col = str(cols_cfg.get("frame", "Frame"))
    x_col = str(cols_cfg.get("x", "X"))
    y_col = str(cols_cfg.get("y", "Y"))
    z_col = str(cols_cfg.get("z", "Z"))

    rename_cfg = (com_cfg.get("rename", {}) or {}) if isinstance(com_cfg, dict) else {}
    out_x = str(rename_cfg.get("x", "COMx"))
    out_y = str(rename_cfg.get("y", "COMy"))
    out_z = str(rename_cfg.get("z", "COMz"))

    pdf = pd.read_excel(path, skiprows=skiprows, sheet_name=sheet_name, engine="openpyxl")
    pdf.columns = [str(c).strip() for c in pdf.columns]

    missing = [c for c in [frame_col, x_col, y_col, z_col] if c not in pdf.columns]
    if missing:
        raise ValueError(f"COM file missing required columns {missing}: {path}")

    pdf = pdf[[frame_col, x_col, y_col, z_col]].copy()
    pdf[frame_col] = pd.to_numeric(pdf[frame_col], errors="coerce")
    pdf[x_col] = pd.to_numeric(pdf[x_col], errors="coerce")
    pdf[y_col] = pd.to_numeric(pdf[y_col], errors="coerce")
    pdf[z_col] = pd.to_numeric(pdf[z_col], errors="coerce")
    pdf = pdf.dropna(subset=[frame_col])

    df = pl.from_pandas(pdf, include_index=False)
    df = df.rename({frame_col: "Frame", x_col: out_x, y_col: out_y, z_col: out_z})
    df = df.with_columns(
        [
            pl.col("Frame").cast(pl.Int64, strict=False),
            pl.col(out_x).cast(pl.Float64, strict=False),
            pl.col(out_y).cast(pl.Float64, strict=False),
            pl.col(out_z).cast(pl.Float64, strict=False),
        ]
    )
    return df


def _attach_mocap_frame(
    df: pl.DataFrame,
    start_mocap: int,
    end_mocap: int,
    logger: logging.Logger | None,
) -> pl.DataFrame:
    if "Frame" not in df.columns:
        raise ValueError("COM dataframe missing 'Frame'")

    frame_min = df.select(pl.col("Frame").min()).item()
    frame_max = df.select(pl.col("Frame").max()).item()
    if frame_min is None or frame_max is None:
        raise ValueError("COM dataframe has empty Frame column")

    frame_min_i = int(frame_min)
    frame_max_i = int(frame_max)
    expected_len = int(end_mocap - start_mocap + 1)

    if (frame_min_i >= (start_mocap - 5)) and (frame_max_i <= (end_mocap + 5)):
        out = df.rename({"Frame": "MocapFrame"}).with_columns(pl.col("MocapFrame").cast(pl.Int64, strict=False))
        return out

    if frame_min_i in (0, 1):
        observed_len = int(frame_max_i - frame_min_i + 1)
        if abs(observed_len - expected_len) > 5:
            _log(
                logger,
                logging.WARNING,
                f"COM Frame length mismatch (expected≈{expected_len}, observed={observed_len}); mapping anyway",
            )
        offset = frame_min_i
    else:
        _log(
            logger,
            logging.WARNING,
            f"COM Frame does not look like MocapFrame; treating as local index starting at {frame_min_i}",
        )
        offset = frame_min_i

    return (
        df.with_columns((pl.col("Frame") - pl.lit(int(offset)) + pl.lit(int(start_mocap))).alias("MocapFrame"))
        .drop("Frame")
        .with_columns(pl.col("MocapFrame").cast(pl.Int64, strict=False))
    )


def _pad_com_tail_one_mocap_frame_linear(
    df: pl.DataFrame,
    *,
    start_mocap: int,
    end_mocap: int,
    com_cols: list[str],
    logger: logging.Logger | None,
    source_path: Path | None = None,
) -> tuple[pl.DataFrame, bool]:
    if df.is_empty():
        return df, False
    if "MocapFrame" not in df.columns:
        return df, False

    com_cols = [c for c in com_cols if c in df.columns]
    if not com_cols:
        return df, False

    mm = df.select(pl.col("MocapFrame").min().alias("_min"), pl.col("MocapFrame").max().alias("_max")).row(0, named=True)
    min_frame = mm.get("_min")
    max_frame = mm.get("_max")
    if min_frame is None or max_frame is None:
        return df, False

    min_frame_i = int(min_frame)
    max_frame_i = int(max_frame)
    if end_mocap - max_frame_i != 1:
        return df, False

    prev_frame_i = max_frame_i - 1
    if prev_frame_i < start_mocap or min_frame_i > prev_frame_i:
        return df, False

    last_rows = df.filter(pl.col("MocapFrame") == pl.lit(max_frame_i)).select(com_cols)
    prev_rows = df.filter(pl.col("MocapFrame") == pl.lit(prev_frame_i)).select(com_cols)
    if last_rows.height == 0 or prev_rows.height == 0:
        return df, False

    last = last_rows.row(0, named=True)
    prev = prev_rows.row(0, named=True)

    new_values: dict[str, float | None] = {}
    for c in com_cols:
        v_last = last.get(c)
        v_prev = prev.get(c)
        if v_last is None:
            new_values[c] = None
        elif v_prev is None:
            new_values[c] = float(v_last)
        else:
            new_values[c] = float(v_last) + (float(v_last) - float(v_prev))

    out_cols = list(df.columns)
    row_data: dict[str, list[object]] = {c: [None] for c in out_cols}
    row_data["MocapFrame"] = [int(end_mocap)]
    for c, v in new_values.items():
        row_data[c] = [v]

    padded = pl.concat([df, pl.DataFrame(row_data)], how="vertical", rechunk=True).sort("MocapFrame")

    tag = f" ({source_path.name})" if source_path is not None else ""
    _log(logger, logging.INFO, f"COM tail 1-frame linear extrapolated to MocapFrame={end_mocap}{tag}")
    return padded, True


def _interpolate_com_linear_deviceframe(
    df: pl.DataFrame,
    *,
    config: dict,
    group_keys: list[str],
    com_cols: list[str],
    logger: logging.Logger | None,
) -> pl.DataFrame:
    if not com_cols:
        return df
    if "DeviceFrame" not in df.columns:
        _log(logger, logging.WARNING, "COM interpolate requested but DeviceFrame column is missing; skip")
        return df

    com_cols = [c for c in com_cols if c in df.columns]
    if not com_cols:
        return df

    frame_ratio = get_frame_ratio(config)
    if frame_ratio <= 1:
        return df

    device = pl.col("DeviceFrame").cast(pl.Int64, strict=False)
    frac = (device % pl.lit(int(frame_ratio))).cast(pl.Float64) / pl.lit(float(frame_ratio))

    updates: list[pl.Expr] = []
    for c in com_cols:
        v0 = pl.col(c).cast(pl.Float64, strict=False)
        v1 = v0.sort_by("DeviceFrame").shift(-int(frame_ratio)).over(group_keys)

        updates.append(
            (
                pl.when(v0.is_not_null() & v1.is_not_null())
                .then(v0 + (v1 - v0) * frac)
                .when(v0.is_not_null())
                .then(v0)
                .otherwise(v1)
            ).alias(c)
        )

    return df.with_columns(updates) if updates else df


def _apply_com_zeroed_columns_deviceframe(
    df: pl.DataFrame,
    *,
    group_keys: list[str],
    zero_base_cols: list[str],
    zero_suffix: str,
    pre_frames_dev: int,
) -> pl.DataFrame:
    if pre_frames_dev <= 0:
        return df
    if "DeviceFrame" not in df.columns:
        return df

    base_cols = [c for c in zero_base_cols if c in df.columns]
    if not base_cols:
        return df

    pre_mask = pl.col("DeviceFrame").cast(pl.Int64, strict=False) < pl.lit(int(pre_frames_dev))
    baselines = (
        df.filter(pre_mask)
        .group_by(group_keys)
        .agg([pl.col(c).cast(pl.Float64, strict=False).mean().alias(f"_com_base__{c}") for c in base_cols])
    )
    out = df.join(baselines, on=group_keys, how="left")

    updates: list[pl.Expr] = []
    drop_cols: list[str] = []
    for c in base_cols:
        b = f"_com_base__{c}"
        drop_cols.append(b)
        updates.append(
            pl.when(pl.col(b).is_not_null())
            .then(pl.col(c).cast(pl.Float64, strict=False) - pl.col(b).cast(pl.Float64, strict=False))
            .otherwise(None)
            .alias(f"{c}{zero_suffix}")
        )

    if updates:
        out = out.with_columns(updates)
    if drop_cols:
        out = out.drop([c for c in drop_cols if c in out.columns])
    return out


def build_com_table_for_join(
    config: dict,
    trial_info: pl.DataFrame,
    logger: logging.Logger | None = None,
) -> pl.DataFrame:
    com_cfg = (config.get("com", {}) or {}) if isinstance(config, dict) else {}
    if not bool(com_cfg.get("enabled", False)):
        return pl.DataFrame()

    raw_root = Path(str(com_cfg.get("raw_data_root") or "")).expanduser()
    file_regex = str(com_cfg.get("file_regex") or "").strip()
    if not file_regex:
        raise ValueError("config.yaml com.file_regex is required when com.enabled=true")

    index = _index_com_files(raw_root, file_regex, logger)
    if not index:
        _log(logger, logging.INFO, "No COM files found; COM merge will be skipped.")
        return pl.DataFrame()

    seg_cfg = (config.get("segmentation", {}) or {}) if isinstance(config, dict) else {}
    frame_ratio = get_frame_ratio(config)
    pre_frames_dev = int(seg_cfg.get("pre_frames", 1000))
    post_frames_dev = int(seg_cfg.get("post_frames", 1000))

    pre_mocap = int(round(pre_frames_dev / frame_ratio))
    post_mocap = int(round(post_frames_dev / frame_ratio))
    if pre_frames_dev % frame_ratio != 0 or post_frames_dev % frame_ratio != 0:
        _log(
            logger,
            logging.WARNING,
            f"segmentation.pre_frames/post_frames not divisible by frame_ratio={frame_ratio}; using rounded mocap frames",
        )

    required = {"subject", "velocity", "trial_num", "platform_onset", "platform_offset"}
    missing = sorted(required - set(trial_info.columns))
    if missing:
        raise ValueError(f"trial_info missing required columns for COM merge: {missing}")

    cols_to_take = ["subject", "velocity", "trial_num", "platform_onset", "platform_offset"]
    if "date" in trial_info.columns:
        cols_to_take.insert(1, "date")

    meta = trial_info.select(cols_to_take).unique(maintain_order=False)
    meta = meta.with_columns(
        [
            pl.col("subject").cast(pl.Utf8),
            pl.col("velocity").cast(pl.Float64, strict=False),
            pl.col("trial_num").cast(pl.Int64, strict=False),
            pl.col("platform_onset").cast(pl.Int64, strict=False),
            pl.col("platform_offset").cast(pl.Int64, strict=False),
        ]
    )
    if "date" in meta.columns:
        meta = meta.with_columns(pl.col("date").cast(pl.Utf8))

    metas: list[ComTrialMeta] = []
    for rr in meta.iter_rows(named=True):
        onset = rr.get("platform_onset")
        offset = rr.get("platform_offset")
        if onset is None or offset is None:
            continue
        subject = str(rr["subject"])
        date = str(rr.get("date")) if rr.get("date") is not None else None
        velocity = float(rr["velocity"])
        trial_num = int(rr["trial_num"])
        metas.append(
            ComTrialMeta(
                subject=subject,
                date=date,
                velocity=velocity,
                trial_num=trial_num,
                start_mocap=int(onset) - int(pre_mocap),
                end_mocap=int(offset) + int(post_mocap),
            )
        )

    if not metas:
        _log(logger, logging.INFO, "No valid trial metadata for COM merge; skipping.")
        return pl.DataFrame()

    rename_cfg = (com_cfg.get("rename", {}) or {}) if isinstance(com_cfg, dict) else {}
    com_cols = [
        str(rename_cfg.get("x", "COMx")),
        str(rename_cfg.get("y", "COMy")),
        str(rename_cfg.get("z", "COMz")),
    ]

    com_frames: list[pl.DataFrame] = []
    missing_n = 0
    padded_n = 0
    for meta_row in metas:
        vel_key = _velocity_key(meta_row.velocity)
        chosen_path: Path | None = None
        if meta_row.date is not None:
            chosen_path = index.get((meta_row.subject, meta_row.date, vel_key, meta_row.trial_num))
        if chosen_path is None:
            candidates = [
                p
                for (s, _d, v, t), p in index.items()
                if (s, v, t) == (meta_row.subject, vel_key, meta_row.trial_num)
            ]
            if len(candidates) == 1:
                chosen_path = candidates[0]

        if chosen_path is None:
            missing_n += 1
            continue

        try:
            df = _read_com_excel(chosen_path, com_cfg)
            df = _attach_mocap_frame(df, meta_row.start_mocap, meta_row.end_mocap, logger)
            df, did_pad = _pad_com_tail_one_mocap_frame_linear(
                df,
                start_mocap=meta_row.start_mocap,
                end_mocap=meta_row.end_mocap,
                com_cols=com_cols,
                logger=logger,
                source_path=chosen_path,
            )
            if did_pad:
                padded_n += 1
        except Exception as exc:
            _log(logger, logging.WARNING, f"Failed to load COM file {chosen_path}: {exc}")
            continue

        df = df.with_columns(
            [
                pl.lit(meta_row.subject).cast(pl.Utf8).alias("subject"),
                pl.lit(_velocity_key(meta_row.velocity)).cast(pl.Float64).alias("velocity"),
                pl.lit(int(meta_row.trial_num)).cast(pl.Int64).alias("trial_num"),
            ]
        )
        com_frames.append(df)

    if missing_n:
        _log(logger, logging.INFO, f"COM files missing for {missing_n} trials (out of {len(metas)}); continuing without them.")
    if padded_n:
        _log(logger, logging.INFO, f"COM tail 1-frame linear extrapolation applied for {padded_n} trials.")

    if not com_frames:
        _log(logger, logging.INFO, "No COM data loaded; COM merge will be skipped.")
        return pl.DataFrame()

    com_df = pl.concat(com_frames, how="vertical", rechunk=True)
    join_keys = ["subject", "velocity", "trial_num", "MocapFrame"]
    com_df = com_df.unique(subset=join_keys, keep="first")
    return com_df


def _lp_filter_fp_pl(
    df: pl.DataFrame,
    cutoff: float,
    order: int,
    cols: List[str],
    sample_rate_hz: float,
) -> pl.DataFrame:
    sr = float(sample_rate_hz)
    nyq = sr * 0.5
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    padlen = 3 * (max(len(a), len(b)) - 1)

    out = df.clone()
    for col_name in cols:
        if col_name not in out.columns:
            continue
        series = out.get_column(col_name).cast(pl.Float64, strict=False)
        try:
            series = series.fill_nan(None)
        except Exception:
            pass
        try:
            series = series.interpolate()
        except Exception:
            pass
        try:
            series = series.fill_null(strategy="forward").fill_null(strategy="backward")
        except Exception:
            pass
        x = series.to_numpy()
        if len(x) <= padlen:
            continue
        if not np.isfinite(x).all():
            continue
        try:
            filtered = filtfilt(b, a, x.astype(float))
            if not np.isfinite(filtered).all():
                continue
            out = out.with_columns(pl.Series(name=col_name, values=filtered))
        except Exception:
            pass
    return out


class EMGProcessor:
    def __init__(self, config: dict, option_config: dict):
        sig_config = config["signal_processing"]
        self.sample_rate = get_device_hz(config)
        self.pad_frames = option_config["pad_frames"]
        self.high_pass = option_config["high_pass"]
        self.low_pass = option_config["low_pass"]
        self.enable_demeaning = option_config.get("enable_demeaning", True)
        self.enable_rectification = option_config.get("enable_rectification", True)

        if not isinstance(self.sample_rate, (int, float)) or self.sample_rate <= 0:
            raise ValueError(f"Invalid sample_rate: {self.sample_rate}. Must be positive number.")
        if not isinstance(self.pad_frames, int) or self.pad_frames < 0:
            raise ValueError(f"Invalid pad_frames: {self.pad_frames}. Must be non-negative integer.")

        self.actual_pad = 0

    def _apply_filter(self, data: np.ndarray, cutoff: float, order: int, btype: str) -> np.ndarray:
        if not isinstance(data, np.ndarray) or data.size == 0:
            raise ValueError("Input data must be non-empty numpy array")
        if not isinstance(order, int) or order <= 0:
            raise ValueError(f"Filter order must be positive integer, got {order}")
        if not isinstance(cutoff, (int, float)) or cutoff <= 0:
            raise ValueError(f"Cutoff frequency must be positive, got {cutoff}")
        if btype not in ["low", "high", "band", "bandstop"]:
            raise ValueError(f"Invalid filter type: {btype}")

        nyq = 0.5 * float(self.sample_rate)
        if cutoff >= nyq:
            raise ValueError(f"Cutoff frequency {cutoff} Hz is >= Nyquist frequency {nyq} Hz")

        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)
        filtered_data = np.copy(data)
        for col in range(data.shape[1]):
            filtered_data[:, col] = filtfilt(b, a, data[:, col])
            if not np.isfinite(filtered_data[:, col]).all():
                raise ValueError(f"NaN/Inf values detected after {btype}-pass filtering")
        return filtered_data

    def process(self, data: np.ndarray) -> np.ndarray:
        self.actual_pad = min(self.pad_frames, max(int(data.shape[0] * 0.25), 10))
        padded = np.concatenate(
            (np.zeros((self.actual_pad, data.shape[1])), data, np.zeros((self.actual_pad, data.shape[1]))),
            axis=0,
        )

        high_filtered = self._apply_filter(padded, self.high_pass["cutoff"], self.high_pass["order"], "high")

        processed = high_filtered
        if self.enable_demeaning:
            processed = processed - np.mean(processed, axis=0)
        if self.enable_rectification:
            processed = np.abs(processed)

        low_filtered = self._apply_filter(processed, self.low_pass["cutoff"], self.low_pass["order"], "low")
        final_data = low_filtered[self.actual_pad : -self.actual_pad, :] if self.actual_pad > 0 else low_filtered
        return np.clip(final_data, 0, None)


def _process_single_cycle_wrapper(args):
    cycle_data_tuple, config, emg_cols, option_config = args
    try:
        (subject, velocity, trial_num), cycle_data = cycle_data_tuple

        key_cols = ["subject", "velocity", "trial_num"]
        if any(col not in cycle_data.columns for col in key_cols):
            raise ValueError(f"Missing required key columns: {key_cols}")
        if cycle_data.select(key_cols).unique().height != 1:
            raise ValueError("Trial mixing detected in Stage02 input")

        # Ensure deterministic time order within trial regardless of upstream global sorting
        if "DeviceFrame" in cycle_data.columns:
            cycle_data = cycle_data.sort("DeviceFrame")
        elif "original_DeviceFrame" in cycle_data.columns:
            cycle_data = cycle_data.sort("original_DeviceFrame")

        emg_np = cycle_data.select(emg_cols).to_numpy().astype(np.float64)
        if emg_np.shape[0] < 20:
            logger.warning(f"Skipping trial {subject}_V{velocity}_T{trial_num}: only {emg_np.shape[0]} frames")
            return None

        processor = EMGProcessor(config, option_config)
        processed_emg_data = processor.process(emg_np)
        processed_emg_df = pl.DataFrame(processed_emg_data, schema=emg_cols)

        metadata_df = cycle_data.drop(emg_cols)

        fp_cols_cfg = config.get("forceplate", {}).get("merge_columns", [])
        cop_cfg = (config.get("forceplate", {}) or {}).get("cop", {}) or {}
        cop_cols_cfg = cop_cfg.get("columns")
        if cop_cols_cfg is None:
            cop_cols_cfg = ["Cx", "Cy", "Cz"]
        elif isinstance(cop_cols_cfg, list):
            cop_cols_cfg = [str(c).strip() for c in cop_cols_cfg]
        else:
            raise ValueError("forceplate.cop.columns는 list[str] 이어야 합니다.")

        fp_base_cols = [col for col in fp_cols_cfg if col in metadata_df.columns]
        cop_base_cols = [col for col in cop_cols_cfg if col in fp_base_cols]
        fp_cols = [col for col in fp_base_cols if col not in cop_base_cols]

        fp_zero_cfg = config.get("forceplate", {}).get("zeroed", {}) or {}
        if bool(fp_zero_cfg.get("enabled", False)):
            zero_cols_cfg = fp_zero_cfg.get("columns")
            if zero_cols_cfg is None:
                base_cols = list(fp_cols_cfg)
            elif isinstance(zero_cols_cfg, list):
                base_cols = [str(c).strip() for c in zero_cols_cfg]
            else:
                raise ValueError("forceplate.zeroed.columns는 list[str] 이어야 합니다.")
            suffix = str(fp_zero_cfg.get("suffix", "_zero"))
            for c in (c for c in base_cols if c not in cop_base_cols):
                z = f"{c}{suffix}"
                if z in metadata_df.columns:
                    fp_cols.append(z)
        if fp_cols:
            fp_config = config.get("forceplate_processing", {}).get("low_pass", {"cutoff": 20, "order": 4})
            fp_cutoff = float(fp_config.get("cutoff", 20))
            fp_order = int(fp_config.get("order", 4))
            device_hz = get_device_hz(config)
            metadata_df = _lp_filter_fp_pl(metadata_df, fp_cutoff, fp_order, fp_cols, device_hz)

        required_for_cop = {"Fz", "Mx", "My"}
        if required_for_cop.issubset(set(metadata_df.columns)):
            cop_thr = float(cop_cfg.get("fz_threshold_n", 20.0))

            fz = pl.col("Fz").cast(pl.Float64, strict=False)
            mx = pl.col("Mx").cast(pl.Float64, strict=False)
            my = pl.col("My").cast(pl.Float64, strict=False)
            ok_fz = fz.is_finite() & (fz.abs() >= pl.lit(cop_thr))

            metadata_df = metadata_df.with_columns(
                [
                    pl.when(ok_fz).then((-my) / fz).otherwise(None).alias("Cx"),
                    pl.when(ok_fz).then(mx / fz).otherwise(None).alias("Cy"),
                    pl.lit(0.0).cast(pl.Float64).alias("Cz"),
                ]
            )

            if bool(fp_zero_cfg.get("enabled", False)) and "DeviceFrame" in metadata_df.columns:
                pre_frames = int((config.get("segmentation", {}) or {}).get("pre_frames", 1000))
                suffix = str(fp_zero_cfg.get("suffix", "_zero"))
                pre_mask = pl.col("DeviceFrame").cast(pl.Int64, strict=False) < pl.lit(pre_frames)

                metadata_df = metadata_df.with_columns(
                    [
                        pl.col("Cx").cast(pl.Float64, strict=False).filter(pre_mask).mean().alias("_cop_base_cx"),
                        pl.col("Cy").cast(pl.Float64, strict=False).filter(pre_mask).mean().alias("_cop_base_cy"),
                    ]
                ).with_columns(
                    [
                        pl.when(pl.col("_cop_base_cx").is_not_null())
                        .then(pl.col("Cx").cast(pl.Float64, strict=False) - pl.col("_cop_base_cx"))
                        .otherwise(None)
                        .alias(f"Cx{suffix}"),
                        pl.when(pl.col("_cop_base_cy").is_not_null())
                        .then(pl.col("Cy").cast(pl.Float64, strict=False) - pl.col("_cop_base_cy"))
                        .otherwise(None)
                        .alias(f"Cy{suffix}"),
                    ]
                )
                drop_cols = [c for c in ["_cop_base_cx", "_cop_base_cy"] if c in metadata_df.columns]
                if drop_cols:
                    metadata_df = metadata_df.drop(drop_cols)

        num_rows = min(len(metadata_df), len(processed_emg_df))
        return pl.concat(
            [metadata_df.slice(0, num_rows), processed_emg_df.slice(0, num_rows)],
            how="horizontal",
        )
    except Exception as e:
        key_str = (
            f"{subject}_V{velocity}_T{trial_num}"
            if "subject" in locals() and "velocity" in locals() and "trial_num" in locals()
            else "unknown_trial"
        )
        logger.error(f"Error processing trial {key_str}: {e}\n{traceback.format_exc()}")
        return None


class StageRunner:
    def __init__(self, input_path: Path, output_dir: Path, config_path: str, debug: bool = False):
        self.input_path = input_path
        self.output_dir = output_dir
        self.config_path = config_path
        if debug:
            logger.setLevel(logging.DEBUG)

        self.config = self._load_configuration()

    def _load_configuration(self):
        log_and_print("\nLoading configuration...")
        config = load_config_yaml(self.config_path)
        log_and_print(f"[OK] Configuration loaded successfully from {self.config_path}")

        required_keys = ["signal_processing", "muscles"]
        for key in required_keys:
            if key not in config:
                log_and_print(f"[WARNING] Missing required config key: {key}", logging.WARNING)

        try:
            _ = get_device_hz(config)
        except Exception as exc:
            log_and_print(f"[CRITICAL] Invalid sampling config: {exc}", logging.CRITICAL)
            raise
        return config

    def _find_input_file(self) -> Path:
        parquet_name = (
            self.config.get("pipeline_files", {}).get("stage01_merged_dataset")
            or "merged_data_comprehensive.parquet"
        )

        log_and_print("\nLooking for input parquet from Stage 01 ...")

        # Treat explicit *.parquet paths as file candidates even if they don't exist yet.
        # This avoids mistakenly appending the parquet name again.
        if self.input_path.suffix.lower() == ".parquet" or self.input_path.is_file():
            target_parquet = self.input_path
        else:
            target_parquet = self.input_path / parquet_name

        log_and_print(f"  Checking: {target_parquet} {'[OK]' if target_parquet.exists() else '[ERROR]'}")
        if target_parquet.exists():
            log_and_print(f"[OK] Using input file: {target_parquet}")
            return target_parquet

        msg = f"Stage02 requires Stage01 output parquet. File not found: {target_parquet}"
        log_and_print(f"[CRITICAL] {msg}", logging.CRITICAL)
        raise FileNotFoundError(msg)

    def _get_processing_options(self) -> List[Dict[str, Any]]:
        sig_config = self.config["signal_processing"]
        if sig_config.get("processing_options"):
            return sig_config["processing_options"]

        return [
            {
                "name": "default",
                "description": "Legacy single processing option",
                "high_pass": sig_config["high_pass"],
                "low_pass": sig_config["low_pass"],
                "pad_frames": sig_config["pad_frames"],
                "enable_demeaning": sig_config.get("enable_demeaning", True),
                "enable_rectification": sig_config.get("enable_rectification", True),
            }
        ]

    def execute(self) -> bool:
        log_and_print("\n" + "=" * 60 + "\nStarting Stage 02: EMG Filtering\n" + "=" * 60)
        log_and_print(f"Input path: {self.input_path}")
        log_and_print(f"Output directory: {self.output_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        input_file = self._find_input_file()

        log_and_print(f"\n[DATA] Loading segmented data from {input_file.name}...")
        data = read_parquet_robust(input_file, logger)

        muscle_names = self.config["muscles"]["names"]
        emg_cols = [col for col in muscle_names if col in data.columns]
        metadata_cols = [col for col in data.columns if col not in emg_cols]

        group_keys = ["subject", "velocity", "trial_num"]
        missing_keys = [k for k in group_keys if k not in data.columns]
        if missing_keys:
            raise ValueError(f"Stage02 missing required grouping keys: {missing_keys}")

        trial_meta_cols = ["subject", "velocity", "trial_num", "platform_onset", "platform_offset"]
        if "date" in data.columns:
            trial_meta_cols.insert(1, "date")
        trial_info = data.select([c for c in trial_meta_cols if c in data.columns]).unique()
        com_table = build_com_table_for_join(self.config, trial_info, logger=logger)
        com_cfg = self.config.get("com", {}) or {}
        com_interp_enabled = bool(com_cfg.get("interpolate", False))
        rename_cfg = (com_cfg.get("rename", {}) or {}) if isinstance(com_cfg, dict) else {}
        com_raw_cols = [
            str(rename_cfg.get("x", "COMx")),
            str(rename_cfg.get("y", "COMy")),
            str(rename_cfg.get("z", "COMz")),
        ]
        com_raw_cols = [c for c in com_raw_cols if c in com_table.columns] if com_table.height > 0 else []

        zero_cfg = (com_cfg.get("zeroed", {}) or {}) if isinstance(com_cfg, dict) else {}
        zero_enabled = bool(zero_cfg.get("enabled", False))
        zero_suffix = str(zero_cfg.get("suffix", "_zero"))
        cols_cfg = zero_cfg.get("columns")
        if cols_cfg is None:
            zero_base_cols = list(com_raw_cols)
        elif isinstance(cols_cfg, list):
            zero_base_cols = [str(c).strip() for c in cols_cfg]
        else:
            raise ValueError("com.zeroed.columns는 list[str] 이어야 합니다.")

        com_zero_cols = [f"{c}{zero_suffix}" for c in zero_base_cols] if zero_enabled else []

        com_out_cols = com_raw_cols + com_zero_cols
        if com_table.height > 0 and com_raw_cols:
            log_and_print(
                f"[OK] Loaded COM table: {com_table.height} rows, cols={com_raw_cols} "
                f"(interpolate={com_interp_enabled}, zeroed={zero_enabled})"
            )
        else:
            log_and_print("[INFO] No COM data loaded (or COM disabled); skipping COM merge.")

        seg_cfg = (self.config.get("segmentation", {}) or {}) if isinstance(self.config, dict) else {}
        pre_frames_dev = int(seg_cfg.get("pre_frames", 1000))

        parts = data.partition_by(group_keys, maintain_order=True, as_dict=True)
        sorted_keys = sorted(parts.keys(), key=lambda k: (k[0], float(k[1]), int(k[2])))
        cycle_groups = [((k[0], float(k[1]), int(k[2])), parts[k]) for k in sorted_keys]
        total_cycles = len(cycle_groups)
        log_and_print(f"[OK] Prepared {total_cycles} trials with {len(emg_cols)} EMG channels for processing.")

        processing_options = self._get_processing_options()
        log_and_print(f"\n[CONFIG] Found {len(processing_options)} processing options.")

        n_jobs = min(16, os.cpu_count() or 1)
        log_and_print(f"[WORKERS] Using {n_jobs} parallel workers for processing.")
        trialgroup_size = 5000

        all_option_results: Dict[str, pl.DataFrame | None] = {}
        for i, option_config in enumerate(processing_options, 1):
            option_name = option_config["name"]
            log_and_print(f"\n--- Processing Option {i}/{len(processing_options)}: {option_name} ---")

            processed_chunks: List[pl.DataFrame] = []
            processed_count = 0
            for start in range(0, total_cycles, trialgroup_size):
                chunk = cycle_groups[start : start + trialgroup_size]
                args_list = [(cycle, self.config, emg_cols, option_config) for cycle in chunk]

                results = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(_process_single_cycle_wrapper)(args) for args in args_list
                )

                dfs = [df for df in results if isinstance(df, pl.DataFrame) and df.height > 0]
                if dfs:
                    processed_chunks.append(pl.concat(dfs, how="vertical", rechunk=True))

                processed_count += len(chunk)
                log_and_print(f"processed {processed_count} / {total_cycles} trials", logging.INFO, "02_emg_filtering")

            if not processed_chunks:
                log_and_print(
                    f"[ERROR] No cycles were successfully processed with option '{option_name}'",
                    logging.ERROR,
                )
                all_option_results[option_name] = None
                continue

            processed_df = pl.concat(processed_chunks, how="vertical", rechunk=True).select(metadata_cols + emg_cols)

            if com_table.height > 0 and com_out_cols:
                processed_df = processed_df.with_columns(
                    pl.col("velocity").cast(pl.Float64, strict=False).round(6).alias("__velocity_key")
                )
                com_join = (
                    com_table.with_columns(pl.col("velocity").cast(pl.Float64, strict=False).round(6).alias("__velocity_key"))
                    .drop(["velocity"])
                )
                processed_df = processed_df.join(
                    com_join,
                    on=["subject", "trial_num", "MocapFrame", "__velocity_key"],
                    how="left",
                ).drop(["__velocity_key"])

                if com_interp_enabled:
                    processed_df = _interpolate_com_linear_deviceframe(
                        processed_df,
                        config=self.config,
                        group_keys=group_keys,
                        com_cols=com_raw_cols,
                        logger=logger,
                    )

                if zero_enabled:
                    processed_df = _apply_com_zeroed_columns_deviceframe(
                        processed_df,
                        group_keys=group_keys,
                        zero_base_cols=zero_base_cols,
                        zero_suffix=zero_suffix,
                        pre_frames_dev=pre_frames_dev,
                    )

                ordered_cols = metadata_cols + [c for c in com_out_cols if c not in metadata_cols] + emg_cols
                ordered_cols = [c for c in ordered_cols if c in processed_df.columns]
                processed_df = processed_df.select(ordered_cols)

            option_dir = self.output_dir / option_name
            option_dir.mkdir(parents=True, exist_ok=True)

            output_name = (
                self.config.get("pipeline_files", {}).get("stage02_processed_emg")
                or "processed_emg_data.parquet"
            )
            save_parquet(processed_df, option_dir / output_name)
            log_and_print(f"[OK] Saved {len(processed_df)} records for option '{option_name}'")

            all_option_results[option_name] = processed_df

        self._save_summary(input_file, all_option_results)
        log_and_print("\n" + "=" * 60 + "\nStage 02: EMG Filtering completed successfully!\n" + "=" * 60)
        return True

    def _save_summary(self, input_file: Path, results: Dict[str, pl.DataFrame | None]):
        summary_file = self.output_dir / "processing_summary.txt"
        options = self._get_processing_options()
        selected_option = self.config["signal_processing"].get("selected_option", options[0]["name"])

        with open(summary_file, "w", encoding="utf-8-sig") as f:
            f.write("===== Stage 02: EMG Filtering Summary =====\n")
            f.write(f"Input file: {input_file.name}\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write("\n===== Processing Options Summary =====\n")
            f.write(f"Number of options: {len(options)}\n")
            f.write(f"Selected for downstream: {selected_option}\n")
            f.write(f"Sample rate: {get_device_hz(self.config)} Hz\n")

            for i, opt in enumerate(options, 1):
                f.write(
                    f"\n--- Option {i}: {opt['name']} {'[SELECTED]' if opt['name'] == selected_option else ''} ---\n"
                )
                f.write(f"High-pass: {opt['high_pass']['cutoff']} Hz (order: {opt['high_pass']['order']})\n")
                f.write(f"Low-pass: {opt['low_pass']['cutoff']} Hz (order: {opt['low_pass']['order']})\n")
                f.write(f"Padding: {opt['pad_frames']} frames\n")
                f.write(
                    f"Demeaning: {opt.get('enable_demeaning', True)}, Rectification: {opt.get('enable_rectification', True)}\n"
                )
                if results.get(opt["name"]) is not None:
                    f.write(f"Output: Success, {len(results[opt['name']])} records generated.\n")
                else:
                    f.write("Output: Processing failed for this option.\n")

        log_and_print(f"\n[STATS] Summary saved to: {summary_file}")
