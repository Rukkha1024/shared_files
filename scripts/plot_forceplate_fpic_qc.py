#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scr.config_helpers import MODULE_DIR, load_config_yaml, get_output_path


def _resolve_default_paths() -> tuple[Path, Path]:
    cfg = load_config_yaml()
    out_dir = get_output_path("01_dataset", "")
    diag_name = cfg.get("pipeline_files", {}).get("stage01_forceplate_subtract_diagnostics") or "forceplate_subtract_diagnostics.parquet"
    cop_qc_dir_name = cfg.get("pipeline_files", {}).get("stage01_forceplate_cop_qc_dir") or "cop_qc"
    return (out_dir / diag_name, out_dir / cop_qc_dir_name)


def _safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in s)


def _velocity_match_atol(cfg: dict[str, Any]) -> float:
    return float((cfg.get("forceplate", {}) or {}).get("velocity_match_atol", 1e-9))


def _velocity_isclose_expr(velocity: float, *, atol: float) -> pl.Expr:
    return (pl.col("velocity") - float(velocity)).abs() <= float(atol)


def _velocity_filter_expr(velocities: list[float], *, atol: float) -> pl.Expr:
    if not velocities:
        return pl.lit(True)
    exprs = [_velocity_isclose_expr(v, atol=atol) for v in velocities]
    out = exprs[0]
    for expr in exprs[1:]:
        out = out | expr
    return out


def _velocity_isclose_np(values: np.ndarray, velocity: float, *, atol: float) -> np.ndarray:
    return np.isclose(np.asarray(values, dtype=float), float(velocity), atol=float(atol), rtol=0.0)


def _configure_korean_font() -> None:
    from matplotlib import font_manager as fm

    candidates = [
        Path("/mnt/c/Windows/Fonts/malgun.ttf"),
        Path("/mnt/c/Windows/Fonts/NanumGothic.ttf"),
        Path("/mnt/c/Windows/Fonts/NotoSansKR-VF.ttf"),
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            fm.fontManager.addfont(str(p))
            name = fm.FontProperties(fname=str(p)).get_name()
            matplotlib.rcParams["font.family"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue


def _stage1_safe_filename(s: str) -> str:
    bad = '<>:"/\\|?*\n\r\t'
    out = "".join("_" if c in bad else c for c in s)
    out = out.strip().strip(".")
    return out or "unit"


def _resolve_stage1_paths(cfg: dict[str, Any]) -> tuple[Path, Path, Path]:
    src_path = Path((cfg.get("perturb_parquet", {}) or {}).get("path") or "")
    if not src_path.is_absolute():
        src_path = (MODULE_DIR / src_path).resolve()

    s1_cfg = (cfg.get("forceplate", {}) or {}).get("stage1_corrections", {}) or {}
    corr_path = s1_cfg.get("corrections_csv")
    if corr_path:
        corr_csv = Path(str(corr_path))
        if not corr_csv.is_absolute():
            corr_csv = (MODULE_DIR / corr_csv).resolve()
    else:
        out_dir = get_output_path("01_dataset", "")
        rel = (cfg.get("pipeline_files", {}) or {}).get("stage01_forceplate_stage1_corrections_csv") or "stage1/corrections.csv"
        corr_csv = (out_dir / rel).resolve()

    plot_dir_cfg = (cfg.get("paths", {}) or {}).get("stage1_plot_dir")
    if plot_dir_cfg:
        stage1_plot_dir = Path(plot_dir_cfg)
        if not stage1_plot_dir.is_absolute():
            stage1_plot_dir = (MODULE_DIR / stage1_plot_dir).resolve()
    else:
        out_dir = get_output_path("01_dataset", "")
        stage1_dir_name = cfg.get("pipeline_files", {}).get("stage01_forceplate_stage1_plot_dir") or "stage1"
        stage1_plot_dir = (out_dir / stage1_dir_name).resolve()

    return src_path, corr_csv, stage1_plot_dir


def _plot_stage1(cfg: dict[str, Any]) -> None:
    import pyarrow.dataset as ds
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    src_parquet, corr_csv, stage1_plot_dir = _resolve_stage1_paths(cfg)
    if not corr_csv.exists():
        raise FileNotFoundError(f"missing `{corr_csv}`; stage1 plot 입력이 없습니다.")
    if not src_parquet.exists():
        raise FileNotFoundError(f"missing `{src_parquet}`; perturb parquet 입력이 없습니다.")

    out_grid_dir = stage1_plot_dir / "grid"
    out_force_dir = stage1_plot_dir / "force"
    out_grid_dir.mkdir(parents=True, exist_ok=True)
    out_force_dir.mkdir(parents=True, exist_ok=True)

    plot_cfg = cfg.get("plots", {}).get("stage1", {}) or {}
    only_corrected_units = bool(plot_cfg.get("only_corrected_units", True))
    max_units = int(plot_cfg.get("max_units", 0) or 0)
    overwrite = bool(plot_cfg.get("overwrite", False))
    velocity_atol = _velocity_match_atol(cfg)

    corr = pd.read_csv(corr_csv)
    corr["subject"] = corr["subject"].astype(str)
    corr["velocity"] = corr["velocity"].astype(float)
    corr["trial_num"] = corr["trial_num"].astype(int)
    corr = corr.sort_values(["subject", "velocity", "trial_num"], kind="mergesort")

    if only_corrected_units and "correction_applied" in corr.columns:
        corr = corr[corr["correction_applied"] == True].copy()

    if len(corr) == 0:
        raise RuntimeError("no target units to plot (check config plots.stage1.only_corrected_units)")

    input_mtime = max(float(corr_csv.stat().st_mtime), float(src_parquet.stat().st_mtime))

    cop_min_abs_fz_n = (
        (cfg.get("stage1", {}) or {}).get("thresholds", {}) or {}
    ).get("fz_low_thr_n")
    if cop_min_abs_fz_n is None:
        cop_min_abs_fz_n = (cfg.get("forceplate", {}) or {}).get("cop", {}) or {}
        cop_min_abs_fz_n = cop_min_abs_fz_n.get("fz_threshold_n", 20.0)
    cop_min_abs_fz_n = float(cop_min_abs_fz_n)

    dataset = ds.dataset(str(src_parquet), format="parquet")

    def compute_cop_xy(mx_nm: np.ndarray, my_nm: np.ndarray, fz_n: np.ndarray, *, min_abs_fz_n: float) -> tuple[np.ndarray, np.ndarray]:
        fz = fz_n.astype(np.float64, copy=False)
        mx = mx_nm.astype(np.float64, copy=False)
        my = my_nm.astype(np.float64, copy=False)
        cop_x = np.full_like(fz, np.nan, dtype=np.float64)
        cop_y = np.full_like(fz, np.nan, dtype=np.float64)
        ok = np.isfinite(fz) & np.isfinite(mx) & np.isfinite(my) & (np.abs(fz) >= float(min_abs_fz_n))
        if np.any(ok):
            cop_x[ok] = -my[ok] / fz[ok]
            cop_y[ok] = mx[ok] / fz[ok]
        return cop_x, cop_y

    # -------- grid plot (repo1: bw_window_correction_grid_plot.py) --------
    grid_index_rows: list[dict[str, Any]] = []
    plotted_grid = 0
    for r in corr.itertuples(index=False):
        subject = str(r.subject)
        velocity = float(r.velocity)
        trial_num = int(r.trial_num)
        onset = int(r.onset)
        offset = int(r.offset)
        shift_n = float(getattr(r, "shift_n", 0.0))
        applied = bool(getattr(r, "correction_applied", True))

        fname = _stage1_safe_filename(f"{subject}_v{velocity:g}_t{trial_num:03d}_on{onset}_off{offset}_grid.png")
        out_path = out_grid_dir / fname
        is_fresh = out_path.exists() and (float(out_path.stat().st_mtime) >= float(input_mtime))
        if is_fresh and not overwrite:
            grid_index_rows.append(
                {
                    "subject": subject,
                    "velocity": velocity,
                    "trial_num": trial_num,
                    "onset": onset,
                    "offset": offset,
                    "shift_n": shift_n,
                    "correction_applied": applied,
                    "plot_path": str(out_path),
                    "skipped_existing": True,
                    "cop_min_abs_fz_n": float(cop_min_abs_fz_n),
                }
            )
            continue

        filt = (ds.field("subject") == subject) & (ds.field("trial_num") == trial_num)
        tbl = dataset.to_table(filter=filt, columns=["subject", "velocity", "trial_num", "MocapFrame", "Fz", "Mx", "My"])
        if tbl.num_rows == 0:
            continue
        df = tbl.to_pandas()
        df = df[_velocity_isclose_np(df["velocity"], velocity, atol=velocity_atol)].copy()
        if len(df) == 0:
            continue
        df = df.sort_values("MocapFrame", kind="mergesort")
        seg = df[(df["MocapFrame"] >= onset) & (df["MocapFrame"] <= offset)].copy()
        if len(seg) == 0:
            continue

        mocap = seg["MocapFrame"].to_numpy(np.int64, copy=False)
        fz_raw = seg["Fz"].to_numpy(np.float64, copy=False)
        mx_raw = seg["Mx"].to_numpy(np.float64, copy=False)
        my_raw = seg["My"].to_numpy(np.float64, copy=False)

        if applied:
            fz_cor = fz_raw + float(shift_n)
            mx_cor = mx_raw + float(getattr(r, "shift_mx_nm", 0.0))
            my_cor = my_raw + float(getattr(r, "shift_my_nm", 0.0))
        else:
            fz_cor = fz_raw
            mx_cor = mx_raw
            my_cor = my_raw

        copx_raw, copy_raw = compute_cop_xy(mx_raw, my_raw, fz_raw, min_abs_fz_n=cop_min_abs_fz_n)
        copx_cor, copy_cor = compute_cop_xy(mx_cor, my_cor, fz_cor, min_abs_fz_n=cop_min_abs_fz_n)
        copmag_raw = np.sqrt(copx_raw**2 + copy_raw**2)
        copmag_cor = np.sqrt(copx_cor**2 + copy_cor**2)

        fig = plt.figure(figsize=(14, 9), constrained_layout=True)
        gs = fig.add_gridspec(2, 2)

        ax_fz_raw = fig.add_subplot(gs[0, 0])
        ax_fz_cor = fig.add_subplot(gs[1, 0], sharex=ax_fz_raw)

        gs_cop_raw = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.25)
        ax_cop_raw_xy = fig.add_subplot(gs_cop_raw[0, 0])
        ax_cop_raw_mag = fig.add_subplot(gs_cop_raw[0, 1], sharex=ax_fz_raw)

        gs_cop_cor = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], wspace=0.25)
        ax_cop_cor_xy = fig.add_subplot(gs_cop_cor[0, 0])
        ax_cop_cor_mag = fig.add_subplot(gs_cop_cor[0, 1], sharex=ax_fz_raw)

        title = f"{subject} | v={velocity:g} | trial={trial_num} | onset={onset} offset={offset} | shift={shift_n:.1f}N | applied={applied}"
        fig.suptitle(title)

        ax_fz_raw.plot(mocap, fz_raw, linewidth=1.0)
        ax_fz_raw.set_title("Fz (raw) onset~offset")
        ax_fz_raw.set_ylabel("N")
        ax_fz_raw.grid(True, alpha=0.3)

        ax_fz_cor.plot(mocap, fz_cor, linewidth=1.0)
        ax_fz_cor.set_title("Fz (corrected) onset~offset")
        ax_fz_cor.set_ylabel("N")
        ax_fz_cor.set_xlabel("MocapFrame (100Hz)")
        ax_fz_cor.grid(True, alpha=0.3)

        ax_cop_raw_xy.plot(copx_raw, copy_raw, linewidth=1.0)
        ax_cop_raw_xy.set_title("COP (raw) 2D trajectory")
        ax_cop_raw_xy.set_xlabel("COPx")
        ax_cop_raw_xy.set_ylabel("COPy")
        ax_cop_raw_xy.grid(True, alpha=0.3)
        ax_cop_raw_xy.set_aspect("equal", adjustable="box")

        ax_cop_cor_xy.plot(copx_cor, copy_cor, linewidth=1.0)
        ax_cop_cor_xy.set_title("COP (corrected) 2D trajectory")
        ax_cop_cor_xy.set_xlabel("COPx")
        ax_cop_cor_xy.set_ylabel("COPy")
        ax_cop_cor_xy.grid(True, alpha=0.3)
        ax_cop_cor_xy.set_aspect("equal", adjustable="box")

        ax_cop_raw_mag.plot(mocap, copmag_raw, linewidth=1.0)
        ax_cop_raw_mag.set_title("COP (raw) magnitude")
        ax_cop_raw_mag.set_xlabel("MocapFrame (100Hz)")
        ax_cop_raw_mag.set_ylabel("sqrt(x^2+y^2)")
        ax_cop_raw_mag.grid(True, alpha=0.3)

        ax_cop_cor_mag.plot(mocap, copmag_cor, linewidth=1.0)
        ax_cop_cor_mag.set_title("COP (corrected) magnitude")
        ax_cop_cor_mag.set_xlabel("MocapFrame (100Hz)")
        ax_cop_cor_mag.set_ylabel("sqrt(x^2+y^2)")
        ax_cop_cor_mag.grid(True, alpha=0.3)

        for ax in [ax_fz_raw, ax_fz_cor, ax_cop_raw_mag, ax_cop_cor_mag]:
            ax.axvline(onset, linewidth=1.0)
            ax.axvline(offset, linewidth=1.0)

        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        grid_index_rows.append(
            {
                "subject": subject,
                "velocity": velocity,
                "trial_num": trial_num,
                "onset": onset,
                "offset": offset,
                "shift_n": shift_n,
                "correction_applied": applied,
                "plot_path": str(out_path),
                "skipped_existing": False,
                "cop_min_abs_fz_n": float(cop_min_abs_fz_n),
            }
        )
        plotted_grid += 1
        if max_units > 0 and plotted_grid >= max_units:
            break

    if grid_index_rows:
        pd.DataFrame(grid_index_rows).to_csv(out_grid_dir / "index.csv", index=False, encoding="utf-8-sig")

    # -------- force plot (repo1: bw_window_correction_force_plot.py) --------
    force_index_rows: list[dict[str, Any]] = []
    plotted_force = 0
    corr_pl = pl.from_pandas(corr.reset_index(drop=True))
    lf = pl.scan_parquet(str(src_parquet))

    for r in corr_pl.iter_rows(named=True):
        subject = str(r["subject"])
        velocity = float(r["velocity"])
        trial_num = int(r["trial_num"])
        onset = int(r["onset"])
        offset = int(r["offset"])

        shift_fz = float(r.get("shift_n", 0.0) or 0.0)
        shift_fx = float(r.get("shift_fx_n", 0.0) or 0.0)
        shift_fy = float(r.get("shift_fy_n", 0.0) or 0.0)
        shift_mx = float(r.get("shift_mx_nm", 0.0) or 0.0)
        shift_my = float(r.get("shift_my_nm", 0.0) or 0.0)
        shift_mz = float(r.get("shift_mz_nm", 0.0) or 0.0)
        applied = bool(r.get("correction_applied", True))

        fname = _stage1_safe_filename(f"{subject}_v{velocity:g}_t{trial_num:03d}_on{onset}_off{offset}_force.png")
        out_path = out_force_dir / fname
        is_fresh = out_path.exists() and (float(out_path.stat().st_mtime) >= float(input_mtime))
        if is_fresh and not overwrite:
            force_index_rows.append(
                {
                    "subject": subject,
                    "velocity": velocity,
                    "trial_num": trial_num,
                    "onset": onset,
                    "offset": offset,
                    "correction_applied": applied,
                    "shift_n": shift_fz,
                    "shift_fx_n": shift_fx,
                    "shift_fy_n": shift_fy,
                    "shift_mx_nm": shift_mx,
                    "shift_my_nm": shift_my,
                    "shift_mz_nm": shift_mz,
                    "plot_path": str(out_path),
                    "skipped_existing": True,
                }
            )
            continue

        seg = (
            lf.filter(pl.col("subject") == subject)
            .filter(pl.col("trial_num") == trial_num)
            .filter(_velocity_isclose_expr(velocity, atol=velocity_atol))
            .filter((pl.col("MocapFrame") >= onset) & (pl.col("MocapFrame") <= offset))
            .select("MocapFrame", "Fx", "Fy", "Fz", "Mx", "My", "Mz")
            .collect(engine="streaming")
            .sort("MocapFrame")
        )
        if seg.height == 0:
            continue

        mocap = seg["MocapFrame"].to_numpy().astype(np.int64, copy=False)

        fx_raw = seg["Fx"].to_numpy().astype(np.float64, copy=False)
        fy_raw = seg["Fy"].to_numpy().astype(np.float64, copy=False)
        fz_raw = seg["Fz"].to_numpy().astype(np.float64, copy=False)
        mx_raw = seg["Mx"].to_numpy().astype(np.float64, copy=False)
        my_raw = seg["My"].to_numpy().astype(np.float64, copy=False)
        mz_raw = seg["Mz"].to_numpy().astype(np.float64, copy=False)

        if applied:
            fx_cor = fx_raw + float(shift_fx)
            fy_cor = fy_raw + float(shift_fy)
            fz_cor = fz_raw + float(shift_fz)
            mx_cor = mx_raw + float(shift_mx)
            my_cor = my_raw + float(shift_my)
            mz_cor = mz_raw + float(shift_mz)
        else:
            fx_cor = fx_raw
            fy_cor = fy_raw
            fz_cor = fz_raw
            mx_cor = mx_raw
            my_cor = my_raw
            mz_cor = mz_raw

        fig, axs = plt.subplots(3, 2, figsize=(16, 11), constrained_layout=True, sharex=True)
        ax_fx, ax_fy, ax_fz, ax_mx, ax_my, ax_mz = axs.flatten()

        title = (
            f"{subject} | v={velocity:g} | trial={trial_num} | onset={onset} offset={offset} | "
            f"applied={applied} | dFx={shift_fx:.2f}N dFy={shift_fy:.2f}N dFz={shift_fz:.1f}N | "
            f"dMx={shift_mx:.2f}Nm dMy={shift_my:.2f}Nm dMz={shift_mz:.2f}Nm"
        )
        fig.suptitle(title)

        def plot_pair(ax, y_raw, y_cor, label: str, unit: str) -> None:
            ax.plot(mocap, y_raw, linewidth=1.0, label="raw")
            ax.plot(mocap, y_cor, linewidth=1.0, label="corrected")
            ax.set_title(label)
            ax.set_ylabel(unit)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", frameon=False)

        plot_pair(ax_fx, fx_raw, fx_cor, "Fx onset~offset", "N")
        plot_pair(ax_fy, fy_raw, fy_cor, "Fy onset~offset", "N")
        plot_pair(ax_fz, fz_raw, fz_cor, "Fz onset~offset", "N")
        plot_pair(ax_mx, mx_raw, mx_cor, "Mx onset~offset", "N·m")
        plot_pair(ax_my, my_raw, my_cor, "My onset~offset", "N·m")
        plot_pair(ax_mz, mz_raw, mz_cor, "Mz onset~offset", "N·m")

        for ax in axs.flatten():
            ax.set_xlabel("MocapFrame (100Hz)")

        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        force_index_rows.append(
            {
                "subject": subject,
                "velocity": velocity,
                "trial_num": trial_num,
                "onset": onset,
                "offset": offset,
                "correction_applied": applied,
                "shift_n": shift_fz,
                "shift_fx_n": shift_fx,
                "shift_fy_n": shift_fy,
                "shift_mx_nm": shift_mx,
                "shift_my_nm": shift_my,
                "shift_mz_nm": shift_mz,
                "plot_path": str(out_path),
                "skipped_existing": False,
            }
        )

        plotted_force += 1
        if max_units > 0 and plotted_force >= max_units:
            break

    if force_index_rows:
        pl.DataFrame(force_index_rows).sort(["subject", "velocity", "trial_num"]).write_csv(out_force_dir / "index.csv")

    (out_force_dir / "report.md").write_text(
        "\n".join(
            [
                "# Stage1: Fx/Fy/Fz/Mx/My/Mz plots",
                "",
                f"- input parquet: `{src_parquet}`",
                f"- correction table: `{corr_csv}`",
                f"- output dir: `{out_force_dir}`",
                f"- only_corrected_units: {only_corrected_units}",
                f"- plotted(grid/force): {int(plotted_grid)} / {int(plotted_force)}",
                "",
                "## Files",
                "- `grid/index.csv`, `grid/*_grid.png`",
                "- `force/index.csv`, `force/*_force.png`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[ok] stage1 plots → {stage1_plot_dir} (grid={plotted_grid}, force={plotted_force})")


def _finite_stats(x: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if not ok.any():
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "rms": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "abs_max": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }
    v = x[ok]
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if v.size >= 2 else 0.0,
        "rms": float(np.sqrt(np.mean(v**2))),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "abs_max": float(np.max(np.abs(v))),
        "p05": float(np.quantile(v, 0.05)),
        "p50": float(np.quantile(v, 0.50)),
        "p95": float(np.quantile(v, 0.95)),
        "p99": float(np.quantile(v, 0.99)),
    }


def _series_stats(x: np.ndarray, *, prefix: str) -> dict[str, float | int]:
    base = _finite_stats(x)
    out: dict[str, float | int] = {f"{prefix}_{k}": v for k, v in base.items()}
    dx = np.diff(np.asarray(x, dtype=float))
    step = _finite_stats(dx)
    out.update({f"{prefix}_step_{k}": v for k, v in step.items()})
    return out


def _ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return float("nan")
    return float(num / den)


def _central_quantile_filter(
    copx: np.ndarray,
    copy: np.ndarray,
    *,
    q_low: float,
    q_high: float,
    min_valid_points: int,
) -> tuple[np.ndarray, float, float, float]:
    copx = np.asarray(copx, dtype=float)
    copy = np.asarray(copy, dtype=float)
    valid = np.isfinite(copx) & np.isfinite(copy)
    n_valid = int(valid.sum())
    if n_valid < int(min_valid_points):
        keep = valid.copy()
        removed_frac = 0.0 if n_valid else float("nan")
        return keep, float("nan"), float("nan"), float(removed_frac)

    r = np.sqrt(copx[valid] ** 2 + copy[valid] ** 2)
    ql = float(np.nanquantile(r, float(q_low)))
    qh = float(np.nanquantile(r, float(q_high)))
    keep = valid.copy()
    keep[valid] = (r >= ql) & (r <= qh)
    removed_frac = float(1.0 - (keep[valid].mean()))
    return keep, ql, qh, float(removed_frac)


def _compute_cop_xy(
    df: pd.DataFrame,
    *,
    fz_threshold_n: float,
    my_col: str,
    mx_col: str,
    fz_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fz = df[fz_col].to_numpy(dtype=float)
    ok = np.isfinite(fz) & (np.abs(fz) >= float(fz_threshold_n))
    my = df[my_col].to_numpy(dtype=float)
    mx = df[mx_col].to_numpy(dtype=float)
    copx = np.where(ok, (-my / fz), np.nan)
    copy = np.where(ok, (mx / fz), np.nan)
    return copx, copy, ok


def _cop_metrics(
    copx: np.ndarray,
    copy: np.ndarray,
    *,
    prefix: str,
    max_abs_m: float,
    max_step_m: float,
    outlier_cfg: dict[str, Any] | None,
) -> dict[str, float | int]:
    copx = np.asarray(copx, dtype=float)
    copy = np.asarray(copy, dtype=float)
    valid = np.isfinite(copx) & np.isfinite(copy)
    r = np.sqrt(copx**2 + copy**2)
    out: dict[str, float | int] = {
        f"{prefix}_valid_frac": float(valid.mean()) if valid.size else float("nan"),
        f"{prefix}_abs_over_max_frac": float(np.mean(r[valid] > float(max_abs_m))) if valid.any() else 0.0,
    }
    out.update(_series_stats(r, prefix=f"{prefix}_r"))
    step = np.sqrt(np.diff(copx) ** 2 + np.diff(copy) ** 2)
    out.update({f"{prefix}_step_{k}": v for k, v in _finite_stats(step).items()})
    out[f"{prefix}_step_over_max_frac"] = (
        float(np.mean(step[np.isfinite(step)] > float(max_step_m))) if np.isfinite(step).any() else 0.0
    )

    if outlier_cfg and bool(outlier_cfg.get("enabled", False)):
        q_low = float(outlier_cfg["q_low"])
        q_high = float(outlier_cfg["q_high"])
        min_valid = int(outlier_cfg["min_valid_points"])
        keep, ql, qh, removed = _central_quantile_filter(copx, copy, q_low=q_low, q_high=q_high, min_valid_points=min_valid)
        out[f"{prefix}_outlier_q_low"] = q_low
        out[f"{prefix}_outlier_q_high"] = q_high
        out[f"{prefix}_outlier_r_q_low_m"] = ql
        out[f"{prefix}_outlier_r_q_high_m"] = qh
        out[f"{prefix}_outlier_removed_frac"] = removed
        copx_f = np.where(keep, copx, np.nan)
        copy_f = np.where(keep, copy, np.nan)
        out.update(_cop_metrics(copx_f, copy_f, prefix=f"{prefix}_filt95", max_abs_m=max_abs_m, max_step_m=max_step_m, outlier_cfg=None))
    else:
        out[f"{prefix}_outlier_q_low"] = float("nan")
        out[f"{prefix}_outlier_q_high"] = float("nan")
        out[f"{prefix}_outlier_r_q_low_m"] = float("nan")
        out[f"{prefix}_outlier_r_q_high_m"] = float("nan")
        out[f"{prefix}_outlier_removed_frac"] = float("nan")
    return out


def _plot_trial(
    df: pd.DataFrame,
    *,
    subject: str,
    velocity: float,
    trial: int,
    mocapframe_col: str,
    copx_meas: np.ndarray,
    copy_meas: np.ndarray,
    copx_corr: np.ndarray,
    copy_corr: np.ndarray,
    copx_corr_filt95: np.ndarray,
    copy_corr_filt95: np.ndarray,
    fx_meas: np.ndarray,
    fx_corr: np.ndarray,
    my_meas: np.ndarray,
    my_corr: np.ndarray,
    mz_meas: np.ndarray,
    mz_corr: np.ndarray,
    fz_meas: np.ndarray,
    subtitle_lines: list[str],
    out_path: Path,
    fz_threshold_n: float,
) -> None:
    t = df[mocapframe_col].to_numpy(dtype=float)
    onset = df["onset_frame"].iloc[0] if "onset_frame" in df.columns else np.nan
    offset = df["offset_frame"].iloc[0] if "offset_frame" in df.columns else np.nan

    fig = plt.figure(figsize=(18, 10), dpi=150)
    axs = np.array([fig.add_subplot(2, 4, i + 1) for i in range(8)]).reshape(2, 4)

    def _plot_ts(ax, y_meas, y_corr, *, title, ylabel):
        ax.plot(t, y_meas, color="#777777", linewidth=1.0, label="measured (preprocessed)")
        ax.plot(t, y_corr, color="#1f77b4", linewidth=1.0, label="corrected")
        ax.set_title(title)
        ax.set_xlabel("MocapFrame")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    _plot_ts(axs[0, 0], fx_meas, fx_corr, title="Fx", ylabel="Fx [N]")
    _plot_ts(axs[0, 1], my_meas, my_corr, title="My", ylabel="My [N·m]")
    _plot_ts(axs[0, 2], mz_meas, mz_corr, title="Mz", ylabel="Mz [N·m]")

    ax_fz = axs[0, 3]
    ax_fz.plot(t, fz_meas, color="#2ca02c", linewidth=1.2, label="Fz")
    ax_fz.axhline(float(fz_threshold_n), color="#2ca02c", linestyle=":", linewidth=1.0, alpha=0.5)
    ax_fz.axhline(-float(fz_threshold_n), color="#2ca02c", linestyle=":", linewidth=1.0, alpha=0.5)
    ax_fz.set_title("Fz")
    ax_fz.set_xlabel("MocapFrame")
    ax_fz.set_ylabel("Fz [N]")
    ax_fz.grid(True, alpha=0.3)

    ax_xy = axs[1, 0]
    ax_xy.plot(copx_meas, copy_meas, color="#777777", linewidth=1.0, label="measured (preprocessed)")
    ax_xy.plot(copx_corr, copy_corr, color="#1f77b4", linewidth=1.0, label="corrected")
    ax_xy.plot(copx_corr_filt95, copy_corr_filt95, color="#d62728", linewidth=1.0, label="corrected (95% kept)")
    ax_xy.set_title("COP trajectory (XY)")
    ax_xy.set_xlabel("COPx [m]")
    ax_xy.set_ylabel("COPy [m]")
    ax_xy.axis("equal")
    ax_xy.grid(True, alpha=0.3)

    ax_cx = axs[1, 1]
    ax_cx.plot(t, copx_meas, color="#777777", linewidth=1.0, label="measured (preprocessed)")
    ax_cx.plot(t, copx_corr, color="#1f77b4", linewidth=1.0, label="corrected")
    ax_cx.plot(t, copx_corr_filt95, color="#d62728", linewidth=1.0, label="corrected (95% kept)")
    ax_cx.set_title("COPx")
    ax_cx.set_xlabel("MocapFrame")
    ax_cx.set_ylabel("COPx [m]")
    ax_cx.grid(True, alpha=0.3)

    ax_cy = axs[1, 2]
    ax_cy.plot(t, copy_meas, color="#777777", linewidth=1.0, label="measured (preprocessed)")
    ax_cy.plot(t, copy_corr, color="#1f77b4", linewidth=1.0, label="corrected")
    ax_cy.plot(t, copy_corr_filt95, color="#d62728", linewidth=1.0, label="corrected (95% kept)")
    ax_cy.set_title("COPy")
    ax_cy.set_xlabel("MocapFrame")
    ax_cy.set_ylabel("COPy [m]")
    ax_cy.grid(True, alpha=0.3)

    ax_txt = axs[1, 3]
    ax_txt.axis("off")
    ax_txt.set_title("QC summary")
    ax_txt.text(0.02, 0.98, "\n".join(subtitle_lines), ha="left", va="top", fontsize=10, family="monospace")

    for ax in axs.ravel():
        if ax is ax_xy or ax is ax_txt:
            continue
        if np.isfinite(onset):
            ax.axvline(float(onset), color="k", linestyle="--", linewidth=1.0, alpha=0.4)
        if np.isfinite(offset):
            ax.axvline(float(offset), color="k", linestyle="--", linewidth=1.0, alpha=0.4)

    by_label: dict[str, object] = {}
    for ax in axs.ravel():
        hs, ls = ax.get_legend_handles_labels()
        for h, l in zip(hs, ls, strict=False):
            by_label.setdefault(l, h)
    if by_label:
        fig.legend(list(by_label.values()), list(by_label.keys()), loc="lower center", ncol=5, frameon=False)

    fig.suptitle(f"{subject} | v={velocity:g} | trial={trial:03d}")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot forceplate COP QC outputs.")
    parser.add_argument("--subtract", action="store_true", help="Plot COP QC outputs (subtract diagnostics).")
    parser.add_argument("--stage1", action="store_true", help="Plot stage1 correction plots.")
    parser.add_argument(
        "--velocity",
        nargs="+",
        type=float,
        help="Limit subtract/COP QC plots to specific velocities (use with --subtract).",
    )
    args = parser.parse_args(argv)
    default_mode = not args.subtract and not args.stage1
    do_subtract = args.subtract or default_mode
    if args.velocity and not do_subtract:
        parser.error("--velocity requires --subtract.")
    return args


def main() -> int:
    args = _parse_args()
    cfg = load_config_yaml()
    _configure_korean_font()

    # Stage1 plot (원본 repo: report/stage1/grid + report/stage1/force)
    default_mode = not args.subtract and not args.stage1
    do_stage1 = args.stage1 or default_mode
    do_subtract = args.subtract or default_mode

    if do_stage1:
        _plot_stage1(cfg)
    if not do_subtract:
        return 0

    mocap_hz = float(cfg.get("sampling", {}).get("mocap_hz", 100))
    diag_path, out_base = _resolve_default_paths()
    plots_dir = out_base / "plots"

    if not diag_path.exists():
        raise FileNotFoundError(f"diagnostics parquet not found: {diag_path}")

    fp_cfg = cfg.get("forceplate", {}) or {}
    cop_cfg = fp_cfg.get("cop", {}) or {}
    qc_cfg = cop_cfg.get("qc", {}) or {}
    if not bool(qc_cfg.get("enabled", True)):
        raise RuntimeError("forceplate.cop.qc.enabled is false; nothing to do.")

    fz_thr = float(cop_cfg.get("fz_threshold_n", 20.0))
    max_abs_m = float(qc_cfg.get("max_abs_m", 0.5))
    max_step_m = float(qc_cfg.get("max_step_m", 0.05))
    cop_overwrite = bool(qc_cfg.get("overwrite", True))

    of_cfg = qc_cfg.get("outlier_filter", {}) or {}
    outlier_cfg: dict[str, Any] | None = None
    if bool(of_cfg.get("enabled", False)):
        outlier_cfg = {
            "enabled": True,
            "method": str(of_cfg.get("method", "central_quantile")),
            "q_low": float(of_cfg.get("q_low", 0.025)),
            "q_high": float(of_cfg.get("q_high", 0.975)),
            "apply_to": str(of_cfg.get("apply_to", "corrected_radius")),
            "min_valid_points": int(of_cfg.get("min_valid_points", 10)),
        }

    out_csv = out_base / "cop_qc_summary.csv"
    report_md = out_base / "report.md"
    report_html = out_base / "report.html"
    if (not cop_overwrite) and out_csv.exists() and report_md.exists() and report_html.exists() and plots_dir.exists():
        print(f"[ok] COP QC already exists (skip): {out_base}")
        return 0

    diag = pl.read_parquet(diag_path)
    required = {
        "subject",
        "velocity",
        "trial_num",
        "mocap_idx_local",
        "MocapFrame",
        "Fx_measured_100hz",
        "My_measured_100hz",
        "Mz_measured_100hz",
        "Mx_measured_100hz",
        "Fz_measured_100hz",
        "Fx_corrected_100hz",
        "My_corrected_100hz",
        "Mz_corrected_100hz",
        "onset_local_100hz",
        "offset_local_100hz",
    }
    missing = sorted(required - set(diag.columns))
    if missing:
        raise ValueError(f"diagnostics parquet에 필요한 컬럼이 없습니다: {missing}")

    velocity_atol = _velocity_match_atol(cfg)
    velocity_targets: list[float] = []
    if args.velocity:
        velocity_targets = [float(v) for v in args.velocity]
        diag = diag.filter(_velocity_filter_expr(velocity_targets, atol=velocity_atol))
        if diag.height == 0:
            raise RuntimeError(f"no diagnostics rows match velocities: {velocity_targets}")

    out_base.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    units = diag.select(["subject", "velocity", "trial_num"]).unique().sort(["subject", "velocity", "trial_num"])
    rows: list[dict[str, Any]] = []

    for r in units.iter_rows(named=True):
        subject = str(r["subject"])
        velocity = float(r["velocity"])
        trial = int(r["trial_num"])

        sub_pl = diag.filter(
            (pl.col("subject") == subject) & (pl.col("velocity") == velocity) & (pl.col("trial_num") == trial)
        ).sort("mocap_idx_local")
        if sub_pl.height == 0:
            continue

        df = sub_pl.to_pandas()

        mocap = df["MocapFrame"].to_numpy(dtype=float)
        monotonic = bool(np.all(np.diff(mocap) >= 0)) if mocap.size >= 2 else True
        duration_frames = int(len(df))
        duration_s = float(duration_frames) / float(mocap_hz) if mocap_hz else float("nan")

        onset_local = int(df["onset_local_100hz"].iloc[0])
        offset_local = int(df["offset_local_100hz"].iloc[0])
        onset_local = max(0, min(onset_local, max(0, len(df) - 1)))
        offset_local = max(0, min(offset_local, max(0, len(df) - 1)))
        onset_frame = int(df["MocapFrame"].iloc[onset_local]) if len(df) else 0
        offset_frame = int(df["MocapFrame"].iloc[offset_local]) if len(df) else 0
        df["onset_frame"] = onset_frame
        df["offset_frame"] = offset_frame

        fx_meas = df["Fx_measured_100hz"].to_numpy(dtype=float)
        fx_corr = df["Fx_corrected_100hz"].to_numpy(dtype=float)
        my_meas = df["My_measured_100hz"].to_numpy(dtype=float)
        my_corr = df["My_corrected_100hz"].to_numpy(dtype=float)
        mz_meas = df["Mz_measured_100hz"].to_numpy(dtype=float)
        mz_corr = df["Mz_corrected_100hz"].to_numpy(dtype=float)
        fz_meas = df["Fz_measured_100hz"].to_numpy(dtype=float)

        masked = np.abs(fz_meas) < float(fz_thr)
        masked_frac = float(np.mean(masked)) if masked.size else float("nan")

        copx_meas, copy_meas, _ = _compute_cop_xy(
            df,
            fz_threshold_n=fz_thr,
            my_col="My_measured_100hz",
            mx_col="Mx_measured_100hz",
            fz_col="Fz_measured_100hz",
        )
        copx_corr, copy_corr, _ = _compute_cop_xy(
            df,
            fz_threshold_n=fz_thr,
            my_col="My_corrected_100hz",
            mx_col="Mx_measured_100hz",
            fz_col="Fz_measured_100hz",
        )

        if outlier_cfg is not None and str(outlier_cfg.get("apply_to")) == "corrected_radius":
            keep, _, _, _ = _central_quantile_filter(
                copx_corr,
                copy_corr,
                q_low=float(outlier_cfg["q_low"]),
                q_high=float(outlier_cfg["q_high"]),
                min_valid_points=int(outlier_cfg["min_valid_points"]),
            )
        else:
            keep = np.where(np.isfinite(copx_corr) & np.isfinite(copy_corr), True, False)

        copx_corr_filt95 = np.where(keep, copx_corr, np.nan)
        copy_corr_filt95 = np.where(keep, copy_corr, np.nan)

        stats: dict[str, Any] = {
            "subject": subject,
            "velocity": velocity,
            "trial": trial,
            "mocap_hz": mocap_hz,
            "duration_frames": duration_frames,
            "duration_s": duration_s,
            "onset_frame": onset_frame,
            "offset_frame": offset_frame,
            "mocapframe_monotonic": monotonic,
            "fz_threshold_n": fz_thr,
            "masked_frac": masked_frac,
        }

        for name, arr in [
            ("fx_meas", fx_meas),
            ("fx_corr", fx_corr),
            ("my_meas", my_meas),
            ("my_corr", my_corr),
            ("mz_meas", mz_meas),
            ("mz_corr", mz_corr),
            ("fz_meas", fz_meas),
        ]:
            stats.update(_series_stats(arr, prefix=name))

        stats.update(_cop_metrics(copx_meas, copy_meas, prefix="cop_meas", max_abs_m=max_abs_m, max_step_m=max_step_m, outlier_cfg=None))
        stats.update(_cop_metrics(copx_corr, copy_corr, prefix="cop_corr", max_abs_m=max_abs_m, max_step_m=max_step_m, outlier_cfg=outlier_cfg))

        stats["cop_r_absmax_ratio_corr_meas"] = _ratio(
            float(stats.get("cop_corr_r_abs_max", float("nan"))),
            float(stats.get("cop_meas_r_abs_max", float("nan"))),
        )
        stats["cop_step_absmax_ratio_corr_meas"] = _ratio(
            float(stats.get("cop_corr_step_abs_max", float("nan"))),
            float(stats.get("cop_meas_step_abs_max", float("nan"))),
        )
        stats["fx_absmax_ratio_corr_meas"] = _ratio(
            float(stats.get("fx_corr_abs_max", float("nan"))),
            float(stats.get("fx_meas_abs_max", float("nan"))),
        )
        stats["my_absmax_ratio_corr_meas"] = _ratio(
            float(stats.get("my_corr_abs_max", float("nan"))),
            float(stats.get("my_meas_abs_max", float("nan"))),
        )
        stats["mz_absmax_ratio_corr_meas"] = _ratio(
            float(stats.get("mz_corr_abs_max", float("nan"))),
            float(stats.get("mz_meas_abs_max", float("nan"))),
        )

        plot_name = _safe_filename(f"{subject}_v{velocity:g}_t{trial:03d}.png")
        plot_path = plots_dir / plot_name
        stats["plot_path"] = str(plot_path.relative_to(out_base))

        subtitle_lines = [
            f"mocap_hz={mocap_hz:g}Hz  duration={duration_s:.2f}s ({duration_frames} frames)",
            f"monotonic MocapFrame: {monotonic}",
            f"onset={onset_frame} offset={offset_frame}",
            f"Fz<thr frac: {masked_frac:.3f} (thr={fz_thr:g}N)",
            f"COP r abs max (meas/corr): {stats.get('cop_meas_r_abs_max', float('nan')):.3f}/{stats.get('cop_corr_r_abs_max', float('nan')):.3f} m",
            f"Fx abs max (meas/corr): {stats.get('fx_meas_abs_max', float('nan')):.1f}/{stats.get('fx_corr_abs_max', float('nan')):.1f} N",
            f"My abs max (meas/corr): {stats.get('my_meas_abs_max', float('nan')):.2f}/{stats.get('my_corr_abs_max', float('nan')):.2f} N·m",
            f"Mz abs max (meas/corr): {stats.get('mz_meas_abs_max', float('nan')):.2f}/{stats.get('mz_corr_abs_max', float('nan')):.2f} N·m",
            f"ratios corr/meas (COP r/Fx/My/Mz): {stats.get('cop_r_absmax_ratio_corr_meas', float('nan')):.3g} / {stats.get('fx_absmax_ratio_corr_meas', float('nan')):.3g} / {stats.get('my_absmax_ratio_corr_meas', float('nan')):.3g} / {stats.get('mz_absmax_ratio_corr_meas', float('nan')):.3g}",
        ]

        _plot_trial(
            df,
            subject=subject,
            velocity=velocity,
            trial=trial,
            mocapframe_col="MocapFrame",
            copx_meas=copx_meas,
            copy_meas=copy_meas,
            copx_corr=copx_corr,
            copy_corr=copy_corr,
            copx_corr_filt95=copx_corr_filt95,
            copy_corr_filt95=copy_corr_filt95,
            fx_meas=fx_meas,
            fx_corr=fx_corr,
            my_meas=my_meas,
            my_corr=my_corr,
            mz_meas=mz_meas,
            mz_corr=mz_corr,
            fz_meas=fz_meas,
            subtitle_lines=subtitle_lines,
            out_path=plot_path,
            fz_threshold_n=fz_thr,
        )

        rows.append(stats)

    if not rows:
        raise RuntimeError("no units to plot (diagnostics parquet is empty?)")

    out_df = pd.DataFrame(rows).sort_values(["subject", "velocity", "trial"], kind="mergesort")
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    n_trials = int(len(out_df))

    def _improve_frac(corr_col: str, meas_col: str) -> float:
        if corr_col not in out_df.columns or meas_col not in out_df.columns or n_trials == 0:
            return float("nan")
        return float((out_df[corr_col] < out_df[meas_col]).mean())

    improve = {
        "cop_r_abs_max": _improve_frac("cop_corr_r_abs_max", "cop_meas_r_abs_max"),
        "cop_step_abs_max": _improve_frac("cop_corr_step_abs_max", "cop_meas_step_abs_max"),
        "fx_abs_max": _improve_frac("fx_corr_abs_max", "fx_meas_abs_max"),
        "my_abs_max": _improve_frac("my_corr_abs_max", "my_meas_abs_max"),
        "mz_abs_max": _improve_frac("mz_corr_abs_max", "mz_meas_abs_max"),
    }

    report_md.write_text(
        "\n".join(
            [
                "# Forceplate COP QC 보고서",
                "",
                f"- 입력(diagnostics): `{diag_path}`",
                f"- 출력: `{out_base}`",
                f"- CSV: `{out_csv.name}`",
                "- plots: `plots/`",
                f"- fz_threshold_n: {fz_thr:g}",
                f"- 개선 비율(abs max, corrected<measured): COP(r) {improve['cop_r_abs_max']:.3f}, Fx {improve['fx_abs_max']:.3f}, My {improve['my_abs_max']:.3f}, Mz {improve['mz_abs_max']:.3f}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    worst = out_df.sort_values("cop_r_absmax_ratio_corr_meas", ascending=False).head(10)
    report_html.write_text(
        "\n".join(
            [
                "<!doctype html><meta charset='utf-8'><title>Forceplate QC Report</title>",
                "<style>body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:24px;line-height:1.5}code{background:#f2f2f2;padding:0 4px;border-radius:4px}a{color:#1f77b4;text-decoration:none}</style>",
                "<h1>Forceplate COP QC 보고서</h1>",
                f"<ul><li>입력(diagnostics): <code>{diag_path}</code></li><li>CSV: <a href='cop_qc_summary.csv'><code>cop_qc_summary.csv</code></a></li><li>plots: <code>plots/</code></li></ul>",
                f"<p>개선 비율(abs max, corrected&lt;measured): COP(r) <b>{improve['cop_r_abs_max']:.3f}</b>, Fx {improve['fx_abs_max']:.3f}, My {improve['my_abs_max']:.3f}, Mz {improve['mz_abs_max']:.3f}</p>",
                "<h2>Worst cases</h2>",
                "<ul>",
                *[
                    f"<li>{str(r.subject)} v={float(r.velocity):g} trial={int(r.trial):03d} ratio={float(r.cop_r_absmax_ratio_corr_meas):.6g} <a href='{str(r.plot_path)}'>plot</a></li>"
                    for r in worst.itertuples(index=False)
                ],
                "</ul>",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[ok] wrote COP QC report to {out_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
