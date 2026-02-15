아래 내용은 `stages/01_build_dataset.py`(Unified Stage 01) 기준으로, **현재 파이프라인 실행 시 실제로 타지 않는 로직(=정리 후보)** 과 **`config.yaml`에서 `enabled: false`로 인해 비활성화된 로직(=정리/유지 의사결정 필요)** 을 함께 정리한 결과입니다.

---

## 1) Stage 01의 실제 실행 경로(현재 “사용 중” 로직)

`stages/01_build_dataset.py`는 파이프라인 통합을 위해 **모듈 레벨 `run()`**(약 L2563)에서 `DatasetBuilder`를 만들고, 내부 **`DatasetBuilder.run()`**을 호출합니다. 그 안에서 호출되는 핵심 체인은 다음과 같습니다.

* `DatasetBuilder.__init__()`

  * `_load_muscle_names()`, `_load_platform_timing()`, `_load_platform_metadata()`, `_scan_parquet_data()` 등 초기화
* `DatasetBuilder.run()` (약 L2450대)

  * `build_dataset_lazy()`
  * `apply_platform_offset_override_lazy()` *(기능 플래그 off면 즉시 return)*
  * `merge_platform_metadata_lazy()`
  * `ensure_forceplate_stage1_corrections()` → 필요 시 `_generate_forceplate_stage1_corrections()`
  * `apply_forceplate_stage1_corrections_lazy()`
  * `apply_forceplate_inertial_removal_lazy()` → `apply_forceplate_subtract_lazy()`
  * `apply_forceplate_zeroed_columns_lazy()`  *(“*_zero” 컬럼 생성)*
  * `save_dataset()`  *(Parquet streaming sink)*
  * `_validate_output_lazy()` 등 QC

즉, 현재 Stage 01은 **LazyFrame 기반(스트리밍 저장) 로직이 “정상 경로”** 입니다.

---

## 2) “현재 실행 경로에서 호출되지 않는” 로직(정리 후보)

아래 5개 메서드는 파일에 남아 있지만, **현재 Stage 01의 실행 경로에서 호출되지 않습니다.** (리팩토링 시 제거/격리 후보)

| 구성요소                         | 위치(대략) | 내용                                           | 왜 안 타나                                          |
| ---------------------------- | -----: | -------------------------------------------- | ----------------------------------------------- |
| `process_single_trial()`     |  ~L468 | trial 단위로 DataFrame을 잘라 처리하는 per-trial 처리    | 현재는 `build_dataset_lazy()`로만 전체 처리              |
| `calculate_segment_bounds()` |  ~L420 | onset/offset + pre/post로 start/end 계산(루프 기반) | `build_dataset_lazy()`에서 bounds를 Lazy로 계산       |
| `convert_emg_column_names()` |  ~L442 | `Dev1/ai# → muscle` rename                   | `build_dataset_lazy()` 내부에서 regex rename 수행(중복) |
| `merge_platform_metadata()`  |  ~L904 | eager 방식의 metadata merge                     | 실제 사용은 `merge_platform_metadata_lazy()`         |
| `build_dataset()`            |  ~L619 | `build_dataset_lazy().collect()` 래퍼(메모리 적재)  | 현재 엔트리포인트(run)는 streaming 저장 경로만 사용             |

**권장 정리 방향**

* “Stage 01을 항상 streaming/Lazy로만 운용”할 계획이면: 위 5개는 **삭제하거나 `legacy_code/`로 격리**하는 편이 유지보수에 유리합니다.
* “디버깅용으로 build_dataset()를 남길지”는 선택 사항이지만, 남길 경우에도 나머지 4개는 중복/미사용 성격이 강합니다.

---

## 3) `config.yaml`에서 `enabled: false` 때문에 “실제로 실행되지 않는” 로직

현재 `config.yaml`에서 `false`인 항목은 2개입니다.

### 3.1 `forceplate.zeroing.enabled: false`

* **관련 코드 위치:** `build_dataset_lazy()` 내부(약 L718 근방)에서

  * `forceplate.zeroing.enabled`가 `false`면

    * `fp_cols_cfg = []` → forceplate baseline(mean) 제거 관련 `baseline_exprs`가 사실상 비게 되어 **영점(베이스라인) 보정 로직이 스킵**됩니다.
* 의미: “세그먼트 시작 N프레임 평균을 빼는” 구식 zeroing을 Stage01에서 하지 않도록 막아둔 상태입니다(주석과 일치).

**리팩토링 관점 포인트**

* 코드 상 기본값이 `fp_zero_enabled = bool(fp_cfg.get("enabled", True))`로 되어 있어, **config에서 키를 제거하면 기본적으로 True로 동작**할 수 있습니다.
  → “앞으로도 절대 안 쓸 기능”이면 **기능 자체 제거** 또는 **default를 False로 변경**하는 편이 안전합니다.

### 3.2 `forceplate.subtract.alignment.enabled: false`

* **관련 코드 위치:** `apply_forceplate_subtract_lazy()` (약 L1628 시작) 및 템플릿 적용 루프 내부(약 L2278~2295)
* 이 플래그가 `false`면 다음 로직이 **절대 실행되지 않습니다.**

  * `_subtract_best_lag_xcorr()` 호출 (정의: ~L79, 호출: ~L2282)
  * `_subtract_shift_fill()`로 템플릿을 lag만큼 시프트 (호출: ~L2290~2295)
  * 결과적으로 `lag_frames`는 항상 0, `corr`는 NaN으로 남습니다.

**리팩토링 관점 포인트**

* alignment 기능을 앞으로도 쓰지 않을 거면:

  * `_subtract_best_lag_xcorr`, `_subtract_shift_fill`, `_SubtractXCorrResult` 및 관련 출력 컬럼(`unload_template_lag_frames`, `unload_template_corr`)까지 포함해서 **삭제/정리** 대상입니다.
* 반대로 “가끔 켤 수도 있다”면:

  * 기능은 유지하되, **정확히 어떤 상황에서 필요한지(데이터 drift, timing 불일치 등)** 를 기준으로 문서화하고, 테스트/검증 루틴을 함께 두는 게 좋습니다.

---

## 4) “설정은 읽지만 동작에 영향이 없는” (Deprecated) 키 처리 로직

Stage 01에는 아래처럼 “있어도 무시하고 경고만 출력”하는 키가 있습니다(코드에서 명시적으로 “더 이상 사용되지 않음” 경고).

* `forceplate.inertial_removal.method` → 항상 subtract로 동작 (약 L1388~1392)
* `forceplate.subtract.window_policy` → 항상 `onset_plus_unload_range`로 동작 (약 L1636~1639)
* `forceplate.subtract.alignment.reference` → 항상 Fx 기준 정렬 (약 L1643~1646)

리팩토링 시에는 보통

* **config 스키마에서 해당 키 제거**
* **코드에서도 경고 분기 제거**
  로 정리하는 것이 깔끔합니다.

---

## 5) 추가로 “현재 사실상 미사용”인 기능 플래그(참고)

* `platform_offset_override` 기능:

  * Stage01 실행 경로에서 항상 `apply_platform_offset_override_lazy()`를 호출하긴 하지만,
  * `config.yaml`에 `platform_offset_override.enabled`가 없어서 기본 `False` → **즉시 return**.
  * “앞으로 쓸 계획이 없으면” 호출 자체를 제거해도 됩니다(성능 영향은 매우 미미).

