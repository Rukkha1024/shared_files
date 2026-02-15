from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from src.config_helpers import PROJECT_ROOT, SCRIPTS_DIR
from src.utils import get_logger

logger = get_logger("pipeline.stage_loader")


class StageLoadError(RuntimeError):
    pass


STAGE_NAMES: list[str] = [
    "01_build_dataset",
    "02_emg_filtering",
    "03_post_processing",
]

_module_cache: dict[str, ModuleType] = {}
_function_cache: dict[str, Callable[[], Any]] = {}


def list_available_stages() -> list[str]:
    return list(STAGE_NAMES)


def _resolve_stage_path(stage_name: str) -> Path:
    if stage_name not in STAGE_NAMES:
        raise StageLoadError(
            f"Unsupported stage: {stage_name}. Available stages: {list_available_stages()}"
        )

    stage_path = (SCRIPTS_DIR / f"{stage_name}.py").resolve()
    if not stage_path.is_file():
        raise StageLoadError(f"Stage script not found: {stage_path}")

    return stage_path


def _load_module_from_path(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise StageLoadError(f"Failed to create import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    # Make it visible to any nested imports and help debuggability.
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as e:
        raise StageLoadError(f"Failed to import stage module {module_path}: {e}") from e

    return module


def load_stage_module(stage_name: str) -> ModuleType:
    if stage_name in _module_cache:
        return _module_cache[stage_name]

    stage_path = _resolve_stage_path(stage_name)
    module_name = f"_emg_stage_{stage_name}"

    # Ensure the project root is importable even when invoked from outside.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    logger.debug(f"Loading stage '{stage_name}' from {stage_path}")
    module = _load_module_from_path(module_name, stage_path)
    _module_cache[stage_name] = module
    return module


def get_stage_function(stage_name: str) -> Callable[[], Any]:
    if stage_name in _function_cache:
        return _function_cache[stage_name]

    module = load_stage_module(stage_name)

    stage_function = getattr(module, "run", None)
    if not callable(stage_function):
        stage_function = getattr(module, "main", None)

    if not callable(stage_function):
        raise StageLoadError(f"No callable run() or main() found in stage: {stage_name}")

    _function_cache[stage_name] = stage_function
    return stage_function

