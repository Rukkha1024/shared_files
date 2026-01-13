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
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_helpers import get_output_path, load_config_yaml
from com_helpers import build_com_table_for_join
from utils import get_logger, log_and_print, read_parquet_robust, save_parquet

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

# Set multiprocessing start method for Windows compatibility
if __name__ == "__main__" or "stages" in __name__:
    if sys.platform == "win32":
        multiprocessing.set_start_method("spawn", force=True)


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
        self.sample_rate = sig_config["sample_rate"]
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
            sampling_cfg = config.get("sampling", {}) or {}
            device_hz = sampling_cfg.get("device_hz") or config.get("signal_processing", {}).get("sample_rate", 1000)
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

        sig_config = config.get("signal_processing", {})
        if "sample_rate" not in sig_config:
            log_and_print("[WARNING] Missing signal_processing.sample_rate in config", logging.WARNING)
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
        com_zero_cols = [c for c in com_zero_cols if c in com_table.columns] if com_table.height > 0 else []

        com_out_cols = com_raw_cols + com_zero_cols
        if com_table.height > 0 and com_out_cols:
            log_and_print(f"[OK] Loaded COM table: {com_table.height} rows, cols={com_out_cols}")
        else:
            log_and_print("[INFO] No COM data loaded (or COM disabled); skipping COM merge.")

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
            f.write(f"Sample rate: {self.config['signal_processing']['sample_rate']} Hz\n")

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


def run():
    config = load_config_yaml("config.yaml")
    parquet_name = config.get("pipeline_files", {}).get("stage01_merged_dataset") or "merged_data_comprehensive.parquet"
    runner = StageRunner(
        input_path=get_output_path("01_dataset", parquet_name),
        output_dir=get_output_path("02_processed", ""),
        config_path="config.yaml",
    )
    return runner.execute()


def main():
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
    sys.exit(main())
