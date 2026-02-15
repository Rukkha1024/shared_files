from __future__ import annotations

from typing import Sequence

import polars as pl


def standardize_join_keys_lf(lf: pl.LazyFrame, keys: Sequence[str]) -> pl.LazyFrame:
    """
    Standardize join-key dtypes for robust joins across inputs.

    - subject: Utf8
    - velocity: Float64
    - trial/trial_num: Int64
    """
    casts: list[pl.Expr] = []
    schema = lf.collect_schema()
    schema_names = set(schema.names())

    for key in keys:
        if key not in schema_names:
            continue
        if key == "subject":
            casts.append(pl.col(key).cast(pl.Utf8))
        elif key == "velocity":
            casts.append(pl.col(key).cast(pl.Float64, strict=False))
        elif key in ("trial", "trial_num"):
            casts.append(pl.col(key).cast(pl.Int64, strict=False))

    return lf.with_columns(casts) if casts else lf


def normalize_option_token(value: str) -> str:
    """Normalize option tokens for tolerant matching (e.g., Chvatal -> Chvatal_35-40)."""
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())

