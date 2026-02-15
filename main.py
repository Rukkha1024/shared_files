#!/usr/bin/env python3
"""
Compatibility entrypoint for the EMG analysis pipeline.

Historically, this project was executed via:
  python main.py

As part of the `stages/` → `scripts/` refactor, the canonical runner now lives at:
  scripts/run_pipeline.py

This file remains to avoid breaking existing workflows.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    runner = Path(__file__).parent / "scripts" / "run_pipeline.py"
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()

