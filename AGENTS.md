## Work Procedure
Always follow this procedure when performing tasks:
1. **Plan the changes**: Before making any code modifications, create a detailed plan outlining what will be changed and why
2. **Get user confirmation**: Present the plan to the user and wait for explicit confirmation before proceeding
3. **Modify code**: Make the necessary code changes according to the confirmed plan
4. **Git commit in Korean**: Commit your changes with a Korean commit message
5. **Run the modified code**: Execute the modified code to verify your work

---
## Environment rules
- Use the existing conda env: `module` (WSL2).
- Always run Python/pip as: `conda run -n module python` / `conda run -n module pip`.
- **Do not** create or activate any `venv` or `.venv` or run `uv venv`.
- If a package is missing, prefer:
  1) `mamba/conda install -n module <pkg>` (if available)
  2) otherwise `conda run -n module pip install <pkg>`
- Before running Python, verify the interpreter path with:
  `conda run -n module python -c "import sys; print(sys.executable)"`

--
## Code Rules 
- Always design code for high reusability and central control via `config.yaml`.
- Whenever you see the same logic or configuration emerge in two or more places, refactor it into a `config.yaml` entry for the parameters.
- Before introducing a new constant or parameter in code, first ask: “Should this live in `config.yaml` so it can be centrally managed?” If yes, add it to `config.yaml` and reference it from there.
- Use "polars" then "pandas" library. 
---

### **Codebase Rule: Configuration Management**

#### **Core Principle: Centralized Control**
The primary goal is to centralize shared values across multiple scripts. This ensures consistency and minimizes code modifications when parameters change.

#### **Items to Include in Config Files:**
1.  **Paths and Directories:** Define paths to data, logs, and outputs (e.g., `RAW_DATA_DIR`, `OUTPUT_DIR`).
2.  **File Identification Patterns:** Store regex or fixed strings for parsing filenames (e.g., `VELOCITY_PATTERN`, `TRIAL_PATTERNS`).
3.  **Data Structure Definitions:** List column names for data extraction or processing (e.g., `FORCEPLATE_COLUMNS`).
4.  **Fixed Processing Constants:** Define constants derived from the experimental setup (e.g., `FRAME_RATIO`, `FORCEPLATE_DATA_START`).
5.  **Tunable Analysis Parameters:** Specify parameters that researchers might adjust (e.g., filter cutoffs, normalization methods).
6.  **Shared Texts:** Centralize common log messages or report headers (e.g., `STAGE01_SUMMARY_HEADER`).

#### **Exclusion Rule:**
- **Visualization Settings:** Do not include settings related to the visual appearance of plots (e.g., colors, fonts, line styles). These should be managed within the visualization code itself.

---

# Codebase Rule: Perturbation Task Data Processing (Generalized)

## 1) Primary processing unit
* The minimum, non-negotiable unit for processing, caching, file naming, and grouping is:
  * **`subject-velocity-trial`**
* All intermediate and final artifacts must be generated, stored, and merged at this unit.

## 2) Configuration authority
* `config.yaml` is the single source of truth for:
  * **channels**
  * **sampling rates**
  * **processing parameters**
* The code must not contain hardcoded duplicates of config-defined values. If a value exists in `config.yaml`, the runtime must consume it from `config.yaml`.

## 3) Time axis policy (flexible, explicit)
* A dataset must provide **one usable time index** column (examples: `DeviceFrame`, `MocapFrame`, `Timestamp`).
* Within each `subject-velocity-trial`, the chosen time index must be:
  * **monotonic (non-decreasing)**
* The time index origin may be either:
  * **global/absolute**, or
  * **trial-zero (local)**
* The chosen origin must be **explicitly declared** (via metadata and/or `config.yaml`).


## 4) Sampling-rate relationship and provenance backup
* The sampling rates are part of the data’s provenance and must be configuration-controlled:
  * `MocapFrame` domain: **100 Hz**
  * `DeviceFrame` domain: **1000 Hz**
* The conversion relationship between these domains (e.g., `FRAME_RATIO`) must be derived from `config.yaml` values and must not be hardcoded.
* If any operation **redefines the time axis** (including but not limited to):
  * changing sampling rates,
  * changing the conversion ratio,
  * shifting the origin to onset/offset alignment,
  * normalization/resampling,
  * any transformation that changes how time is interpreted,
    then an immutable, original absolute sensor frame pointer must be preserved as a dedicated column, e.g.:**`original_DeviceFrame`**
* `original_DeviceFrame` must be treated as **read-only provenance**:
  * it must **not be overwritten**,
  * it must **not be replaced by recomputation**.
* If correction/repair is required, create a **new** column (e.g., `repaired_original_DeviceFrame`) and keep the original column intact.


## 5) Derived time axes are allowed (but must not overwrite)
* Alignment (e.g., onset-locked) and normalization/resampling are permitted **only as derived columns**, for example:
  * `aligned_*`, `x_norm`, `resampled_*`
* Derived time axes must **not overwrite** the dataset’s chosen time index.
* The chosen time index column must remain usable for:
  * grouping,
  * validation,
  * traceability.

## 6) Event columns (if present)
* If event columns exist (e.g., `onset`, `offset`), they must:
  * be expressed in a **clearly defined time index domain**, and
  * fall within the run’s time range for that `subject-velocity-trial`.

## 7) Required validation (minimum)
For each dataset/run:

* `subject`, `velocity`, `trial` are present and non-null.
* The chosen time index exists and is monotonic per `subject-velocity-trial`.
* Duplicates in the primary time index within a run are either:
  * absent, or
  * explicitly handled and documented (implementation-defined, but must be deliberate and reproducible).

