
lets do refactoring. 

scripts, src 폴더로만 구분한다. 유저는 코딩에 대해서 잘 모르지만, scripts는  실제로 실행(Run)하는 파일들이 들어갑니다. src에 있는 도구들을 가져와서 실제 데이터를 넣고 돌리는 곳이고  src에 직접 실행하는 파일이 아닙니다. 다른 파일에서 import해서 가져다 쓰기 위한 함수나 클래스 정의들이 들어갑니다.이라는 것으로 알고 있어. 


<구체적 지시사항>
act as you are 안드레 카파시. 너에게 최대한 자율성을 부여한다. 
stage 1,2,3 이 실시하는 기능은 동일하지만 함수나 클래스는 src로 옮겨도 돼. 
</구체적 지시사항> 


<내가 공부한 scripts, src>
데이터 분석 프로젝트에서 `src`와 `scripts`를 나누는 것은 **"재사용 가능한 도구(함수/클래스)"**와 **"실행 명령(작업)"**을 구분하는 것입니다.

데이터 분석은 보통 '데이터 수집 → 전처리 → 모델링 → 시각화'의 과정을 거치는데, 이 과정에서 폴더가 섞이면 나중에 코드를 다시 쓸 때 매우 골치 아파집니다.

---

### 1. `src` 폴더: "나만의 분석 도구함 (라이브러리)"

여기는 **직접 실행하는 파일이 아닙니다.** 다른 파일에서 `import`해서 가져다 쓰기 위한 **함수나 클래스 정의**들이 들어갑니다.

*   **무엇이 들어가는가?**
    *   데이터 전처리 함수 (`remove_outliers`, `fill_missing_values`)
    *   데이터베이스 연결 설정 (`get_db_connection`)
    *   모델 구조 정의 (`MyPredictionModel`)
    *   시각화 스타일 설정 (`plot_setup`)
*   **특징:**
    *   이 안의 코드에는 `print()`나 `plot.show()` 같은 실행 코드가 거의 없습니다.
    *   대신 `def`(함수 정의)나 `class`(클래스 정의)로 가득 차 있습니다.

> **예시 (`src/cleaning.py`)**
> ```python
> # 이 파일은 직접 실행하지 않습니다.
> def remove_nulls(df):
>     """결측치를 제거하는 함수"""
>     return df.dropna()
> 
> def normalize_date(date_str):
>     """날짜 형식을 통일하는 함수"""
>     return pd.to_datetime(date_str)
> ```

---

### 2. `scripts` 폴더: "작업 지시서 (실행 파일)"

여기는 **실제로 실행(`Run`)하는 파일**들이 들어갑니다. `src`에 있는 도구들을 가져와서 실제 데이터를 넣고 돌리는 곳입니다.

*   **무엇이 들어가는가?**
    *   데이터 다운로드 자동화 스크립트
    *   매일 아침 실행해야 하는 리포트 생성 파일
    *   모델 학습(Training) 실행 파일
*   **특징:**
    *   터미널에서 `python scripts/train_model.py` 처럼 직접 실행합니다.
    *   "1번부터 5번까지 순서대로 실행해!" 같은 절차적 코드가 들어갑니다.

> **예시 (`scripts/run_daily_report.py`)**
> ```python
> # 1. src 폴더에 있는 함수를 가져옵니다.
> from src.cleaning import remove_nulls, normalize_date
> import pandas as pd
> 
> # 2. 데이터를 불러옵니다.
> raw_data = pd.read_csv('today_data.csv')
> 
> # 3. src의 도구를 사용해 청소합니다.
> clean_data = remove_nulls(raw_data)
> clean_data['date'] = clean_data['date'].apply(normalize_date)
> 
> # 4. 결과를 저장합니다.
> clean_data.to_csv('report_result.csv')
> print("오늘의 리포트 생성 완료!")
> ```

---

### 3. 실제 폴더 구조 예시 (쇼핑몰 데이터 분석)

쇼핑몰에서 **"다음 달 이탈할 고객 예측"** 프로젝트를 한다고 가정해 봅시다.

```text
my_project/
├── src/                  (도구함: 핵심 로직)
│   ├── __init__.py
│   ├── data_loader.py    # DB에서 데이터 가져오는 함수들
│   ├── preprocessing.py  # 이상치 제거, 문자열 처리 함수들
│   └── models.py         # 머신러닝 모델(XGBoost 등) 설정값
│
├── scripts/              (작업장: 실행 버튼)
│   ├── 01_fetch_data.py  # "데이터 가져와서 저장해!" (실행용)
│   ├── 02_train.py       # "모델 학습 시작해!" (실행용)
│   └── 03_predict.py     # "다음 달 이탈자 명단 뽑아!" (실행용)
│
├── notebooks/            (실험실)
│   └── exploration.ipynb # 주피터 노트북 (낙서장처럼 막 코딩해보는 곳)
│
└── data/                 (창고)
    └── customers.csv
```

### 4. 왜 이렇게 나누나요? (Jupyter Notebook만 쓰면 안 되나요?)

데이터 분석 입문 때는 `Jupyter Notebook` 하나에 모든 코드를 다 때려 넣습니다. 하지만 실무에서는 문제가 생깁니다.

1.  **복붙의 지옥:** 전처리 로직이 바뀌면 `분석_v1.ipynb`, `분석_v2.ipynb`, `최종_진짜최종.ipynb` 파일 10개를 열어서 다 수정해야 합니다.
    *   -> **해결:** 로직을 `src`에 한 번만 작성하고 불러와서 쓰면 `src` 파일 하나만 고치면 됩니다.
2.  **자동화 불가능:** 주피터 노트북은 사람이 `Shift+Enter`를 눌러야 실행됩니다. 매일 새벽 3시에 자동으로 분석을 돌리려면 `.py` 파일(스크립트)이 필요합니다.
    *   -> **해결:** `scripts` 폴더의 파일을 스케줄러(Cron, Airflow)에 등록하면 됩니다.

### 요약
*   **`src`**: **레고 블록** (조립 부품) - "데이터를 깨끗하게 만드는 기능", "그래프 그리는 기능"
*   **`scripts`**: **완성된 레고 설명서** (조립 과정) - "블록 A와 B를 합쳐서 결과물을 만들어라"
<내가 공부한 scripts, src>