"""
Consolidated utility functions for the EMG analysis project.
"""

from __future__ import annotations

import logging
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

import polars as pl
from tqdm import tqdm


def _find_project_root(start: Path | None = None) -> Path:
    """
    Locate the project root directory robustly.

    Heuristic:
    - `config.yaml` exists
    - `scripts/` directory exists

    Falls back to the parent of this file's directory (i.e., `scr/..`).
    """
    start_path = Path(__file__).resolve() if start is None else Path(start).resolve()
    start_dir = start_path if start_path.is_dir() else start_path.parent

    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "config.yaml").is_file() and (candidate / "scripts").is_dir():
            return candidate

    return start_dir.parent


_PROJECT_ROOT = _find_project_root()

# Global set to keep track of initialized loggers to prevent duplicate handlers
_initialized_loggers: set[str] = set()

# Define default config values at module level
_DEFAULT_LOG_LEVEL = logging.INFO
_DEFAULT_LOGS_DIR = _PROJECT_ROOT / "logs"
_DEFAULT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
_DEFAULT_LOG_FILE_PATH = _DEFAULT_LOGS_DIR / "emg_analysis_fallback.log"
_DEFAULT_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_DEFAULT_LOG_BACKUP_COUNT = 3


def get_logger(name: str, level: int = None) -> logging.Logger:
    """
    Configure and return a logger instance with specified name.
    """
    logger = logging.getLogger(name)

    log_level_to_use = _DEFAULT_LOG_LEVEL
    log_file_path_to_use = _DEFAULT_LOG_FILE_PATH
    log_max_bytes_to_use = _DEFAULT_LOG_MAX_BYTES
    log_backup_count_to_use = _DEFAULT_LOG_BACKUP_COUNT

    # Check if logger already initialized
    if name in _initialized_loggers:
        if level is not None and logger.level != level:
            logger.setLevel(level)
        elif level is None and logger.level != log_level_to_use:
            logger.setLevel(log_level_to_use)
        return logger

    # Set logging level
    effective_level = level if level is not None else log_level_to_use
    logger.setLevel(effective_level)

    # Add handlers if not already present
    if not logger.handlers:
        # File Handler
        log_file = Path(log_file_path_to_use)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        fh = RotatingFileHandler(
            log_file,
            maxBytes=log_max_bytes_to_use,
            backupCount=log_backup_count_to_use,
            encoding="utf-8",
        )
        fh_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch_formatter = logging.Formatter("%(levelname)s: %(name)s: %(message)s")
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

    logger.propagate = False
    _initialized_loggers.add(name)
    return logger


def create_output_directory(output_path: Path) -> None:
    """Create output directory if it doesn't exist."""
    output_path.mkdir(parents=True, exist_ok=True)


def save_parquet(
    df: pl.DataFrame | pl.LazyFrame,
    filepath: Path,
    compression: str = "zstd",
    compression_level: int = 10,
    statistics: bool = True,
) -> None:
    """Save Polars DataFrame to Parquet format with standardized compression."""
    logger = get_logger("utils.save_parquet")
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(df, pl.LazyFrame):
        df.sink_parquet(
            path,
            compression=compression,
            compression_level=compression_level,
            statistics=statistics,
        )
        logger.info(f"Saved parquet (lazy): {path}")
        return

    if not isinstance(df, pl.DataFrame):
        raise TypeError("save_parquet expects a Polars DataFrame or LazyFrame")

    df.write_parquet(
        path,
        compression=compression,
        compression_level=compression_level,
        statistics=statistics,
    )
    logger.info(f"Saved parquet: {path}")


def read_parquet_robust(
    file_path: Path,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> pl.DataFrame:
    """Read Parquet file with basic robustness and logging."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    try:
        df = pl.read_parquet(path, **kwargs)
        if logger:
            logger.debug(f"Successfully read Parquet file: {path.name}")
        return df
    except Exception as e:
        if logger:
            logger.error(f"Failed to read Parquet file: {path} ({e})")
        raise


def log_and_print(message: str, level: int = logging.INFO, logger_name: str = None) -> None:
    """Print a message to the console and log it."""
    print(message)
    if logger_name:
        logger = get_logger(logger_name)
    else:
        # Get logger from calling context if available
        import inspect

        frame = inspect.currentframe()
        caller_frame = frame.f_back
        caller_globals = caller_frame.f_globals
        logger = caller_globals.get("logger")
        if logger is None:
            logger = get_logger("utils")
    logger.log(level, message)


def write_summary(stage_name: str, stats: dict, out_path: str, format_template: str = "default") -> None:
    """Write summary file with standard formatting."""
    import datetime
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8-sig") as f:
        f.write(f"===== {stage_name} Summary =====\n")

        # Add timestamp if not provided
        if "timestamp" not in stats:
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        else:
            f.write(f"Timestamp: {stats['timestamp']}\n")

        # Write main statistics
        for key, value in stats.items():
            if key != "timestamp":
                formatted_key = key.replace("_", " ").title()
                f.write(f"{formatted_key}: {value}\n")

    log_and_print(f"✅ Summary saved to: {out_path}")


def run_parallel(
    worker_function: Callable,
    items_to_process: Iterable[Any],
    max_workers: int,
    description: str = "Processing",
    timeout: Optional[float] = None,
) -> List[Any]:
    """
    Execute worker function with ThreadPoolExecutor for parallel processing.
    Uses ThreadPoolExecutor for Windows compatibility.
    """
    items_list = list(items_to_process)
    num_items = len(items_list)

    if num_items == 0:
        return []

    if max_workers <= 0:
        raise ValueError("max_workers must be greater than 0")

    logger = get_logger(__name__)
    logger.info(f"Starting parallel processing of {num_items} items with {max_workers} workers")

    results: list[Any] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(worker_function, item): item for item in items_list}

        for future in tqdm(as_completed(future_to_item, timeout=timeout), total=num_items, desc=description):
            item = future_to_item[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing item {item}: {e}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Worker function failed for item {item}: {e}")

    logger.info(f"Successfully completed parallel processing of {num_items} items")
    return results
