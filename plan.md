# COM Helper 통합 계획 (Stage 02)

## 목표
`com_helpers.py`의 COM 처리 로직을 `stages/02_emg_filtering.py` 내부로 통합하여 코드베이스 일관성 확보

---

## 현재 상태
- **분산된 구조**: COM 로직이 `com_helpers.py` (shared_files 루트)에 위치
- **의존성**: Stage 02가 외부 helper 파일에 의존
- **문제점**: Stage 코드만으로는 전체 처리 흐름 파악 불가

---

## 작업 단계

### 1. Stage 02 내부 함수 이동
**com_helpers.py에서 02_emg_filtering.py로 이동할 함수들:**
- `_velocity_key()` - velocity 반올림 유틸
- `_log()` - 로깅 헬퍼
- `_index_com_files()` - COM 파일 인덱싱
- `_read_com_excel()` - Excel COM 데이터 읽기
- `_attach_mocap_frame()` - MocapFrame 변환 및 정렬
- `_apply_zeroed_columns()` - Baseline zeroing 처리
- `build_com_table_for_join()` - 메인 함수
- `ComTrialMeta` dataclass

**위치**: Stage 02 파일 상단 (import 섹션 직후)

### 2. Import 정리
- `from com_helpers import build_com_table_for_join` 제거
- COM 관련 import는 함수 내부에 유지 (pandas, openpyxl)

### 3. 기존 com_helpers.py 처리
**옵션 A**: 파일 삭제 후 git commit
**옵션 B**: `Archive/` 폴더로 이동 (백업 유지)

### 4. 검증
- Stage 02 단독 실행 테스트
- COM 병합 결과 무결성 확인 (기존 출력과 비교)
- Import 에러 없는지 확인

---

## 예상 효과
✅ **단일 파일에서 전체 로직 확인 가능**  
✅ **외부 helper 의존성 제거**  
✅ **stages/ 폴더 내 일관된 구조 유지**  
✅ **코드 유지보수성 향상**  

---

## 주의사항
- COM 관련 config.yaml 설정은 그대로 유지
- 함수 시그니처 및 동작 변경 없음 (단순 이동)
- 기존 출력 결과와 동일성 보장 필요
