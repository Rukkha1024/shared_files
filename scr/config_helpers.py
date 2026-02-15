"""
Configuration helpers for the EMG analysis project.

This module centralizes:
- Project root discovery
- Common path constants (DATA_DIR, OUTPUT_DIR, LOGS_DIR, STAGES_DIR)
- YAML config loading
- Sampling-rate helpers (mocap/device Hz and derived frame ratio)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _find_project_root(start: Path | None = None) -> Path:
    """
    Locate the project root directory robustly.

    Heuristic:
    - `config.yaml` exists
    - `stages/` directory exists

    Falls back to the parent of this file's directory (i.e., `scr/..`).
    """
    start_path = Path(__file__).resolve() if start is None else Path(start).resolve()
    start_dir = start_path if start_path.is_dir() else start_path.parent

    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "config.yaml").is_file() and (candidate / "stages").is_dir():
            return candidate

    # Fallback: `scr/` is expected to live directly under the project root.
    return start_dir.parent


# --- Path configurations ---
# Keep legacy names for backward compatibility, but treat MODULE_DIR as the
# project root rather than the physical directory of this file.
PROJECT_ROOT = _find_project_root()
MODULE_DIR = PROJECT_ROOT
BASE_DIR = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "DATA"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"
STAGES_DIR = PROJECT_ROOT / "stages"

# Ensure key directories exist, especially for outputs/logs.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
STAGES_DIR.mkdir(parents=True, exist_ok=True)


def get_output_path(stage_name: str, filename: str) -> Path:
    """
    Generate a Path for an output file within a specific stage directory.

    Example:
        get_output_path("01_dataset", "merged_data.parquet")
        -> output/01_dataset/merged_data.parquet
    """
    stage_output_dir = OUTPUT_DIR / str(stage_name)
    stage_output_dir.mkdir(parents=True, exist_ok=True)
    return stage_output_dir / str(filename)


def load_config_yaml(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path:
            Path to YAML config file.
            - If None: uses default `config.yaml` in MODULE_DIR (project root).
            - If relative: treated as relative to MODULE_DIR.

    Returns:
        Configuration dictionary (empty dict if file is empty).
    """
    if config_path is None:
        resolved = MODULE_DIR / "config.yaml"
    else:
        p = Path(config_path)
        resolved = p if p.is_absolute() else (MODULE_DIR / p)

    try:
        with open(resolved, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if config is None:
            return {}
        if not isinstance(config, dict):
            raise TypeError(f"config.yaml must be a mapping/dict, got {type(config).__name__}")
        return config
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found: {resolved}") from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML config {resolved}: {e}") from e


def get_mocap_hz(config: dict[str, Any]) -> int:
    sampling_cfg = (config.get("sampling", {}) or {}) if isinstance(config, dict) else {}
    mocap_hz = sampling_cfg.get("mocap_hz")
    if mocap_hz is None:
        raise KeyError("config.yaml missing required key: sampling.mocap_hz")
    mocap_hz = int(mocap_hz)
    if mocap_hz <= 0:
        raise ValueError(f"Invalid sampling.mocap_hz: {mocap_hz} (must be > 0)")
    return mocap_hz


def get_device_hz(config: dict[str, Any]) -> int:
    sampling_cfg = (config.get("sampling", {}) or {}) if isinstance(config, dict) else {}
    device_hz = sampling_cfg.get("device_hz")
    if device_hz is None:
        # Backward compatibility: legacy config used signal_processing.sample_rate
        sig_cfg = (config.get("signal_processing", {}) or {}) if isinstance(config, dict) else {}
        device_hz = sig_cfg.get("sample_rate")

    if device_hz is None:
        raise KeyError(
            "config.yaml missing required key: sampling.device_hz (or legacy signal_processing.sample_rate)"
        )

    device_hz = int(device_hz)
    if device_hz <= 0:
        raise ValueError(f"Invalid sampling.device_hz: {device_hz} (must be > 0)")
    return device_hz


def get_frame_ratio(config: dict[str, Any]) -> int:
    """
    Return integer ratio DeviceFrame:MocapFrame derived from sampling rates.
    """
    device_hz = get_device_hz(config)
    mocap_hz = get_mocap_hz(config)

    ratio_f = float(device_hz) / float(mocap_hz)
    ratio_i = int(round(ratio_f))
    if abs(ratio_f - float(ratio_i)) > 1e-9:
        raise ValueError(
            f"DeviceFrame:MocapFrame ratio must be an integer; got device_hz/mocap_hz={ratio_f} "
            f"(device_hz={device_hz}, mocap_hz={mocap_hz})"
        )
    if ratio_i <= 0:
        raise ValueError(f"Invalid derived frame_ratio: {ratio_i} (device_hz={device_hz}, mocap_hz={mocap_hz})")
    return ratio_i

