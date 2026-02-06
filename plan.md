# Plan: COP & COM Velocity Calculation in Pipeline

## 요약

`main.py` 실행 시 COP(Center of Pressure)와 COM(Center of Mass)의 velocity를 자동으로 계산하여 Stage 02 출력에 포함시킵니다.

## 구현 위치

**Stage 02** (`stages/02_emg_filtering.py`):
- COP와 COM 데이터가 모두 준비된 후 (line 809)
- 최종 저장 전 (line 822)

## 계산 방법

### Finite Difference 방식
- **Interior points**: Central difference `v[i] = (pos[i+1] - pos[i-1]) / (2*dt)`
- **First point**: Forward difference `v[0] = (pos[1] - pos[0]) / dt`
- **Last point**: Backward difference `v[n-1] = (pos[n-1] - pos[n-2]) / dt`

### 샘플링 레이트
- **COP**: 1000Hz → dt = 0.001s
- **COM**: 100Hz → dt = 0.01s

## 생성될 컬럼 (총 6개)

### COP Velocity (3개)
```
Cx_vel, Cy_vel, Cz_vel          # Component velocities (m/s)
```

### COM Velocity (3개)
```
COMx_vel, COMy_vel, COMz_vel    # Component velocities (m/s)
```

**참고**:
- Zeroed velocity는 제외 (수학적으로 raw와 동일: d/dt(Cx - const) = d/dt(Cx))
- Magnitude는 제외 (필요시 나중에 √(x²+y²) 계산 가능)

## 구현 단계

### 1. 함수 추가 (`02_emg_filtering.py:447`)
```python
def _calc_velocity_component(df, col, dt):
    """Calculate velocity for a position column using finite differences"""
    # Central/forward/backward difference implementation
    # Preserves nulls from position data
    # Returns df with new {col}_vel column

def _add_velocity_columns(df, config, grouping_cols):
    """Main function: calculate all velocities per trial"""
    # Groups by [subject, velocity, trial_num]
    # COP velocity: Cx, Cy, Cz (1000Hz, dt=0.001s)
    # COM velocity: COMx, COMy, COMz (100Hz, dt=0.01s)
    # Returns df with 6 new velocity columns
```

### 2. 호출 삽입 (`02_emg_filtering.py:809`)
```python
).drop(["__velocity_key"])

# NEW: Add velocity calculations
log_and_print("[VELOCITY] Calculating COP and COM velocities...")
processed_df = _add_velocity_columns(
    processed_df,
    self.config,
    grouping_cols=group_keys
)
log_and_print("[OK] Velocity columns added")

ordered_cols = metadata_cols + ...
```

### 3. Config 업데이트 (`config.yaml:66`)
```yaml
# --- Velocity calculation settings ---
velocity:
  enabled: true
  cop:
    columns: [Cx, Cy, Cz]        # Raw COP position columns
  com:
    columns: [COMx, COMy, COMz]  # Raw COM position columns
  method: "finite_difference"    # Central/forward/backward difference
```

## Critical Files

1. **`stages/02_emg_filtering.py`**
   - Line 447: 2개 함수 추가 (~70 lines)
   - Line 809: velocity 계산 호출 (~6 lines)
   - Line 811-813: 컬럼 순서 업데이트 (~4 lines)

2. **`config.yaml`**
   - Line 66: velocity 설정 추가 (~8 lines)

## Edge Cases 처리

1. **Missing COM data** (~82% null): velocity도 null 유지
2. **COP invalid regions** (Fz < 20N): 이미 null → velocity도 null
3. **Trial boundaries**: `partition_by([subject, velocity, trial_num])` 사용
4. **Short trials** (< 2 frames): null velocity 반환

## 검증 계획

### Unit Tests (`tests/test_velocity_calculation.py`)
- Constant position → zero velocity
- Linear motion → constant velocity
- Null preservation
- Trial boundary isolation

### Integration Tests (Stage 02 실행 후)
```python
# 1. 컬럼 존재 확인
expected = ["Cx_vel", "Cy_vel", "Cz_vel", "COMx_vel", "COMy_vel", "COMz_vel"]
assert all(col in df.columns for col in expected)

# 2. Null 일관성
assert (df["COMx"].is_null() | df["COMx_vel"].is_null()).all()

# 3. 물리적 타당성
cop_vel = df["Cx_vel"].filter(pl.col("Cx_vel").is_not_null())
assert cop_vel.abs().quantile(0.99) < 20.0  # m/s
```

### Visual Validation
```python
# scripts/validate_velocity.py
# Plot position vs velocity time series
# 3 trials: COP/COM position + velocity 비교
```

## 구현 체크리스트

### 코드 작성
- [ ] `_calc_velocity_component()` 함수 추가 (~40 lines)
- [ ] `_add_velocity_columns()` 함수 추가 (~30 lines)
- [ ] `execute()` 메서드에 velocity 계산 호출 삽입 (~6 lines)
- [ ] 컬럼 순서 업데이트 (velocity 컬럼 포함, ~4 lines)
- [ ] `config.yaml`에 velocity 섹션 추가 (~8 lines)

### 테스트
- [ ] Unit test 파일 생성 (`tests/test_velocity_calculation.py`)
  - Constant position → zero velocity
  - Linear motion → constant velocity
  - Null preservation
  - Trial boundary isolation
- [ ] 소규모 데이터로 Stage 02 테스트 실행 (1 subject)
- [ ] Integration tests:
  - 6개 velocity 컬럼 존재 확인
  - Null 일관성 (COM null → velocity null)
  - 물리적 타당성 (velocity < 20 m/s)
- [ ] Visual validation (`scripts/validate_velocity.py`)
  - Position vs velocity 시계열 플롯

### 최종 검증
- [ ] 전체 파이프라인 실행 (`python main.py`)
- [ ] Output 파일 크기 확인 (~50MB 증가 예상)
- [ ] Git commit: "Stage 02에 COP 및 COM 속도 계산 추가"
