from __future__ import annotations

import logging
import re
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import polars as pl


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


def _index_com_files(raw_root: Path, file_regex: str, logger: logging.Logger | None) -> dict[tuple[str, str, float, int], Path]:
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


def _apply_zeroed_columns(df: pl.DataFrame, com_cfg: dict, pre_mocap_frames: int) -> pl.DataFrame:
    zero_cfg = (com_cfg.get("zeroed", {}) or {}) if isinstance(com_cfg, dict) else {}
    if not bool(zero_cfg.get("enabled", False)):
        return df

    if pre_mocap_frames <= 0:
        return df

    suffix = str(zero_cfg.get("suffix", "_zero"))

    cols_cfg = zero_cfg.get("columns")
    if cols_cfg is None:
        rename_cfg = (com_cfg.get("rename", {}) or {}) if isinstance(com_cfg, dict) else {}
        zero_cols = [
            str(rename_cfg.get("x", "COMx")),
            str(rename_cfg.get("y", "COMy")),
            str(rename_cfg.get("z", "COMz")),
        ]
    elif isinstance(cols_cfg, list):
        zero_cols = [str(c).strip() for c in cols_cfg]
    else:
        raise ValueError("com.zeroed.columns는 list[str] 이어야 합니다.")

    zero_cols = [c for c in zero_cols if c in df.columns]
    if not zero_cols:
        return df

    start_mocap = df.select(pl.col("MocapFrame").min()).item()
    if start_mocap is None:
        return df
    start_mocap = int(start_mocap)

    mask = (pl.col("MocapFrame") >= pl.lit(start_mocap)) & (
        pl.col("MocapFrame") < (pl.lit(start_mocap) + pl.lit(int(pre_mocap_frames)))
    )

    baseline = (
        df.filter(mask)
        .select([pl.col(c).cast(pl.Float64, strict=False).mean().alias(c) for c in zero_cols])
        .row(0, named=True)
    )

    updates: list[pl.Expr] = []
    for c in zero_cols:
        base = baseline.get(c)
        if base is None:
            updates.append(pl.lit(None).cast(pl.Float64).alias(f"{c}{suffix}"))
            continue

        try:
            base_val = float(base)
        except Exception:
            updates.append(pl.lit(None).cast(pl.Float64).alias(f"{c}{suffix}"))
            continue

        if not math.isfinite(base_val):
            updates.append(pl.lit(None).cast(pl.Float64).alias(f"{c}{suffix}"))
            continue

        updates.append((pl.col(c).cast(pl.Float64, strict=False) - pl.lit(base_val)).alias(f"{c}{suffix}"))

    return df.with_columns(updates) if updates else df


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
    frame_ratio = int(seg_cfg.get("frame_ratio", 10))
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

    com_frames: list[pl.DataFrame] = []
    missing_n = 0
    for meta_row in metas:
        vel_key = _velocity_key(meta_row.velocity)
        chosen_path: Path | None = None
        if meta_row.date is not None:
            chosen_path = index.get((meta_row.subject, meta_row.date, vel_key, meta_row.trial_num))
        if chosen_path is None:
            candidates = [p for (s, _d, v, t), p in index.items() if (s, v, t) == (meta_row.subject, vel_key, meta_row.trial_num)]
            if len(candidates) == 1:
                chosen_path = candidates[0]

        if chosen_path is None:
            missing_n += 1
            continue

        try:
            df = _read_com_excel(chosen_path, com_cfg)
            df = _attach_mocap_frame(df, meta_row.start_mocap, meta_row.end_mocap, logger)
            df = _apply_zeroed_columns(df, com_cfg, pre_mocap_frames=int(pre_mocap))
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

    if not com_frames:
        _log(logger, logging.INFO, "No COM data loaded; COM merge will be skipped.")
        return pl.DataFrame()

    com_df = pl.concat(com_frames, how="vertical", rechunk=True)
    join_keys = ["subject", "velocity", "trial_num", "MocapFrame"]
    com_df = com_df.unique(subset=join_keys, keep="first")
    return com_df
