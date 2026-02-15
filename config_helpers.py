"""
Configuration helpers module
Contains path constants and helper functions previously in config.py
"""
from pathlib import Path
import yaml


# --- Path configurations ---
# Determine base, module, and data directories dynamically
MODULE_DIR = Path(__file__).parent.resolve()
BASE_DIR = MODULE_DIR.parent
DATA_DIR = MODULE_DIR / "DATA"

# Ensure key directories exist, especially for outputs
OUTPUT_DIR = MODULE_DIR / "output"
LOGS_DIR = MODULE_DIR / "logs"
STAGES_DIR = MODULE_DIR / "stages"

# Create them if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
STAGES_DIR.mkdir(parents=True, exist_ok=True)


def get_output_path(stage_name: str, filename: str) -> Path:
    """
    Generates a Path object for an output file within a specific stage's output directory.
    Example: get_output_path("00_subject_match", "mapping.parquet")
             -> Module/output/00_subject_match/mapping.parquet
    """
    stage_output_dir = OUTPUT_DIR / stage_name
    stage_output_dir.mkdir(parents=True, exist_ok=True)
    return stage_output_dir / filename


def load_config_yaml(config_path: str = None) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses default config.yaml in MODULE_DIR

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if config_path is None:
        config_path = MODULE_DIR / "config.yaml"
    else:
        config_path = Path(config_path)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML config {config_path}: {e}")
