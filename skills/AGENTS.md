# Non-code Environment Notes

## 2026-02-06
- In this WSL2 environment, `conda run -n module python - <<'PY' ... PY` produced no visible stdout.
- Workaround: write the script to a temporary `.py` file and run it with `conda run -n module python /tmp/<script>.py`.
