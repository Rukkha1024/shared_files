## Work Procedure
Always follow this procedure when performing tasks:
1. **Plan the changes**: Before making any code modifications, create a detailed plan outlining what will be changed and why
2. **Get user confirmation**: Present the plan to the user and wait for explicit confirmation before proceeding
3. **Modify code**: Make the necessary code changes according to the confirmed plan
4. **Git Commit**: Commit changes with a Korean commit message specifically.
5. **Run and Verify**: Execute the code and perform MD5 checksum comparison between new outputs and reference files if pipelines or logic were changed.
6. **Finalize**: Record any non-code environment issues in skills/AGENTS.md and clearly specify which skills were used in the final response.


---
## Environment rules
- Use the existing conda env: `module` (WSL2).
- Always run Python/pip as: `conda run -n module python` / `conda run -n module pip`.

---
## **Codebase Rule: Configuration Management**
- Do not restore or roll back files/code that you did not modify yourself. Never attempt to "fix" or revert changes in files unrelated to your current task, including using `git checkout`.
- Use `polars` then `pandas` library.
- **Leverage Parallel Agent Execution**: In WSL2, multiple agents can run in parallel. Proactively launch multiple independent tasks (search, read, validation) simultaneously to reduce turnaround time.

### **Core Principle: Centralized Control**
The primary goal is to centralize shared values across multiple scripts. This ensures consistency and minimizes code modifications when parameters change.

### **Items to Include in Config Files:**
1.  **Paths and Directories:** Define paths to data, logs, and outputs (e.g., `RAW_DATA_DIR`, `OUTPUT_DIR`).
2.  **File Identification Patterns:** Store regex or fixed strings for parsing filenames (e.g., `VELOCITY_PATTERN`, `TRIAL_PATTERNS`).
3.  **Data Structure Definitions:** List column names for data extraction or processing (e.g., `FORCEPLATE_COLUMNS`, `METADATA_COLS`).
4.  **Fixed Processing Constants:** Define constants derived from the experimental setup (e.g., `FRAME_RATIO`, `FORCEPLATE_DATA_START`).
5.  **Tunable Analysis Parameters:** Specify parameters that researchers might adjust (e.g., filter cutoffs, normalization methods).
6.  **Shared Texts:** Centralize common log messages or report headers (e.g., `STAGE03_SUMMARY_HEADER`).

### **Exclusion Rule:**
- **Visualization Settings:** Do not include settings related to the visual appearance of plots (e.g., colors, fonts, line styles). These should be managed within the visualization code itself.
- **Analysis Notebook Exception:** Under `analysis/`, `.ipynb` files are explicitly allowed to import each other directly for exploratory/statistical workflows.

---

# Pipeline Rules: Perturbation Task (Condensed)

## 1) Keys (do not mix)
- Base unit (cache/filename/group): `subject-velocity-trial`
- EMG event/feature unit: `subject-velocity-trial-emg_channel`

## 2) Onset timing workflow (EMG)
<Current Sequence>
1. Calculate onset timing using TKEO or TH.
2. Override with user's manual values.
</Current Sequence>

- Applicable targets (all trial×channel): `non-TKEO(TH)`, `TKEO-TH`, `TKEO-AGLR`
- Manual values are based on **absolute/original_DeviceFrame (1000 Hz)** and take precedence over algorithm results.

## 3) Time axis & domains
- `original_DeviceFrame`: Absolute provenance (1000 Hz). Never overwrite.
- `DeviceFrame`: `original_DeviceFrame - platform_onset` (based on platform_onset=0).
- Mocap ↔ Device (100 Hz ↔ 1000 Hz) conversion/ratio must be managed via `config.yaml` only (No hardcoding).
- Event domains (absolute vs device) are specified in `config.yaml > windowing.event_domains` (Defaults to absolute if undefined).

## 4) Windowing/event join rules (Prevention of recurrence)
- Event columns referenced in windowing must be **generated + joined before the calculation (e.g., iEMG/RMS)**.
- EMG windowing (iEMG/RMS) **uses per-channel event (trial×channel) as is**.
  - Trial-level reduction (`windowing.channel_event_reduce`) applies **only to trial-level calculations** like CoP/CoM.

## 5) Minimum validation (per run)
- `subject`, `velocity`, `trial_num` non-null
- time index monotonic per `subject-velocity-trial`
- window event values exist and are within the corresponding trial range

