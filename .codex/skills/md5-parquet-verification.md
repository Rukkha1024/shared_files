# MD5 verification notes (Parquet outputs)

## Observation
- Re-running the pipeline (`conda run -n module python main.py`) can produce different MD5 checksums for the same `*.parquet` outputs, even when code/config/data are unchanged.
- This suggests the pipeline outputs are not byte-for-byte deterministic across runs (likely due to parallel execution, floating-point reduction order, or Parquet writer chunking/metadata).

## Implication
- A strict `md5sum` comparison of full Parquet files may report `DIFF` even when the underlying data are equivalent (or only differ in insignificant metadata/ordering).

## Recommended regression-check approach
- Keep `md5sum` snapshots for auditing, but also add a content-level comparison when a true regression check is needed:
  - Compare row counts and schemas.
  - Compare key statistics per `subject-velocity-trial` (min/max/mean for critical signals).
  - If hashing is needed: sort by stable keys, normalize dtypes, and hash selected columns (optionally rounding floats).

