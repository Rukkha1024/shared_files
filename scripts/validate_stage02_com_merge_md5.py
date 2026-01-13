#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from config_helpers import load_config_yaml
from utils import save_parquet


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_subset(in_path: Path, out_path: Path, columns: list[str]) -> str:
    lf = pl.scan_parquet(in_path).select(columns)
    save_parquet(lf, out_path)
    return _md5(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="MD5 compare Stage02 outputs (base columns only).")
    parser.add_argument("--ref", required=True, help="Reference parquet path (before COM merge).")
    parser.add_argument("--new", required=True, help="New parquet path (after COM merge).")
    parser.add_argument("--out-dir", default="output/validation", help="Directory for subset parquet outputs.")
    args = parser.parse_args()

    cfg = load_config_yaml("config.yaml")
    com_cfg = cfg.get("com", {}) or {}
    rename_cfg = (com_cfg.get("rename", {}) or {}) if isinstance(com_cfg, dict) else {}
    com_raw_cols = [
        str(rename_cfg.get("x", "COMx")),
        str(rename_cfg.get("y", "COMy")),
        str(rename_cfg.get("z", "COMz")),
    ]
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
    com_cols = com_raw_cols + com_zero_cols

    ref_path = Path(args.ref)
    new_path = Path(args.new)
    if not ref_path.exists():
        raise FileNotFoundError(ref_path)
    if not new_path.exists():
        raise FileNotFoundError(new_path)

    ref_cols = pl.scan_parquet(ref_path).collect_schema().names()
    new_cols = pl.scan_parquet(new_path).collect_schema().names()
    common_cols = [c for c in ref_cols if (c in new_cols) and (c not in com_cols)]
    if not common_cols:
        raise RuntimeError("No common columns found to compare.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_out = out_dir / "ref_base_columns.parquet"
    new_out = out_dir / "new_base_columns.parquet"

    ref_md5 = _write_subset(ref_path, ref_out, common_cols)
    new_md5 = _write_subset(new_path, new_out, common_cols)

    print(f"[REF] {ref_out} md5={ref_md5}")
    print(f"[NEW] {new_out} md5={new_md5}")
    print(f"[RESULT] {'MATCH' if ref_md5 == new_md5 else 'DIFF'} (columns={len(common_cols)})")

    present_com_cols = [c for c in com_cols if c in new_cols]
    if present_com_cols:
        stats_exprs = [pl.len().alias("n_rows")]
        stats_exprs.extend([pl.col(c).is_not_null().sum().alias(f"{c}_non_null") for c in present_com_cols])
        stats = pl.scan_parquet(new_path).select(stats_exprs).collect(engine="streaming").row(0, named=True)
        summary = ", ".join(f"{c}={int(stats.get(f'{c}_non_null', 0) or 0)}" for c in present_com_cols)
        print(f"[COM] n_rows={int(stats.get('n_rows', 0) or 0)}, {summary}")

    return 0 if ref_md5 == new_md5 else 2


if __name__ == "__main__":
    raise SystemExit(main())
