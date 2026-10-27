# 응용 · 얕은 data / outputs 구조

코드7개는 현재 폴더에, 원본2개는 `data/`, 생성 결과는 `outputs/`에 둡니다. 직접 정의하는 함수를 늘리지 않고 실습별 한 파일에서 진행합니다. 원본100건과 전날의 고정 분할을 사용합니다.

```text
02_응용/
├─ README.md
├─ 01_tree_regression.py … 07_project.py
├─ data/
│  ├─ requests.csv
│  └─ split_manifest.csv
└─ outputs/                  # 실행할 때 생성되는 결과
```

[역할 폴더 README](../README.md)에서 설치를 확인하고 이 폴더를 VS Code로 엽니다. 01~06은 독립 실행합니다. 07~08은 같은 파일입니다. 한 파일이 길어 보여도 `제공`으로 표시한 데이터 읽기·전처리는 재사용 코드입니다. **TODO8곳의 모델·지표 연결에 집중**합니다.

숫자는 train 중앙값, 문자는 train 최빈값과 원핫 인코딩으로 처리합니다. Tree·Forest·XGBoost에는 별도 표준화를 하지 않습니다. 모델마다 `clone(pre)`로 아직 학습하지 않은 새 전처리기를 사용합니다. CV에는 반드시 **전처리 전 입력과 Pipeline 전체**를 넣습니다.

`data/requests.csv`는 원본을 보관하는 곳입니다. CSV를 덮어쓰거나 outputs의 수치를 손으로 고쳐 맞추지 않습니다. 비교 CSV의 원래 정밀도는 유지하고 콘솔만 반올림합니다.


## 실습 01 · 회귀 트리의 평균

**목표:** 분류와 회귀의 잎 출력을 구분합니다.

**열 파일:** `01_tree_regression.py`  
**완성할 위치:** `A01`

- **A01**: 회귀 트리를 학습하고 세 입력을 예측하세요.

**힌트:** 분류 때와 같은 fit·predict 형태를 사용합니다.

```powershell
python 01_tree_regression.py
```

**실제 정답 실행으로 확인한 출력**

```text
분기 기준: 500.0
250·850·1200 예측: [120.0, 720.0, 720.0]
왼쪽 평균: 120.0 오른쪽 평균: 720.0
1200은 학습 범위 밖입니다. 트리도 기존 잎 값으로 예측합니다.
```

**생성 파일:** 없음. 콘솔 출력과 설명을 확인합니다.

**완료 기준:** TODO가 실제 계산·모델 호출에 사용되고, 위 출력 및 단위·행 수가 일치하며 다음 질문에 답할 수 있어야 합니다. 반올림 끝자리 차이만 있다면 검증 버전부터 확인합니다.

- 1200의 예측이 계속 증가하지 않는 이유는 무엇인가요?

## 실습 02 · 깊이와 과적합 표

**목표:** train·valid 차이와 잎 개수를 함께 관찰합니다.

**열 파일:** `02_depth.py`  
**완성할 위치:** `A02`

- **A02**: depth를 실제 트리 깊이로 적용하세요.

**힌트:** 기록하는 depth와 실제 max_depth가 같은지 확인합니다.

```powershell
python 02_depth.py
```

**실제 정답 실행으로 확인한 출력**

```text
depth  train_F1  valid_F1  leaves
    1     0.500     0.286       2
    2     0.500     0.286       3
    3     0.710     0.333       4
    5     0.882     0.500       9
 None     1.000     0.500      14
```

**생성 파일:** `outputs/02_depth.csv`

**완료 기준:** TODO가 실제 계산·모델 호출에 사용되고, 위 출력 및 단위·행 수가 일치하며 다음 질문에 답할 수 있어야 합니다. 반올림 끝자리 차이만 있다면 검증 버전부터 확인합니다.

- 깊이가 증가할 때 train·valid F1이 같은 방향으로 움직이나요?

## 실습 03 · Tree와 Forest 비교

**목표:** 좋아진 점수만 찾지 않고 두 종류의 오류를 설명합니다.

**열 파일:** `03_forest.py`  
**완성할 위치:** `A03`

- **A03**: model을 train으로 학습한 뒤 valid를 예측하세요.

**힌트:** model은 전처리가 포함된 Pipeline입니다.

```powershell
python 03_forest.py
```

**실제 정답 실행으로 확인한 출력**

```text
model  train_F1  valid_F1  FP  FN
  Tree      0.71     0.333   4   4
Forest      0.80     0.222   2   5
```

**생성 파일:** `outputs/03_forest.csv`

**완료 기준:** TODO가 실제 계산·모델 호출에 사용되고, 위 출력 및 단위·행 수가 일치하며 다음 질문에 답할 수 있어야 합니다. 반올림 끝자리 차이만 있다면 검증 버전부터 확인합니다.

- valid 실패 6건 중 몇 건을 놓쳤나요?
- Forest가 항상 Tree보다 낫다고 할 수 있나요?

## 실습 04 · 변환 열과 MDI

**목표:** 원본 4열과 전처리 8열을 구분합니다.

**열 파일:** `04_importance.py`  
**완성할 위치:** `A04`

- **A04**: 전처리의 출력 이름과 Forest의 중요도를 가져오세요.

**힌트:** 이름은 pre.get_feature_names_out(), 값은 model의 feature_importances_입니다.

```powershell
python 04_importance.py
```

**실제 정답 실행으로 확인한 출력**

```text
변환된 특성 수: 8
                feature  importance
       num__input_chars      0.5643
     num__allowed_tools      0.1670
   cat__priority_normal      0.0538
      cat__task_type_qa      0.0468
      cat__priority_low      0.0451
 cat__task_type_summary      0.0428
cat__task_type_research      0.0406
     cat__priority_high      0.0398
중요도는 원인이나 신규 데이터 성능을 보장하지 않습니다.
```

**생성 파일:** `outputs/04_importance.csv`

**완료 기준:** TODO가 실제 계산·모델 호출에 사용되고, 위 출력 및 단위·행 수가 일치하며 다음 질문에 답할 수 있어야 합니다. 반올림 끝자리 차이만 있다면 검증 버전부터 확인합니다.

- priority 관련 열이 여러 개로 나뉘는 이유는 무엇인가요?

## 실습 05 · XGBoost와 지표

**목표:** 학습 목적·진단 지표·선택 지표를 구분합니다.

**열 파일:** `05_xgboost.py`  
**완성할 위치:** `A05`

- **A05**: Pipeline을 학습하고 valid를 예측하세요.

**힌트:** PPT 설정은 이미 제공했습니다. 학습·예측에 test를 넣지 않습니다.

```powershell
python 05_xgboost.py
```

**실제 정답 실행으로 확인한 출력**

```text
클래스 순서: [0, 1]
 train_F1  valid_F1  FP  FN  valid_logloss
    0.774     0.182   4   5          0.855
objective는 학습 목적, logloss는 진단용, 후보 비교의 주 지표는 F1입니다.
```

**생성 파일:** `outputs/05_xgboost.csv`

**완료 기준:** TODO가 실제 계산·모델 호출에 사용되고, 위 출력 및 단위·행 수가 일치하며 다음 질문에 답할 수 있어야 합니다. 반올림 끝자리 차이만 있다면 검증 버전부터 확인합니다.

- F1과 logloss가 같은 방향으로만 움직일까요?

## 실습 06 · 동일 5-fold 비교

**목표:** fold 점수·평균·표준편차를 구분합니다.

**열 파일:** `06_cv.py`  
**완성할 위치:** `A06`

- **A06**: 전처리까지 포함한 model의 fold별 F1을 계산하세요.

**힌트:** 입력은 전처리 전 X_train입니다.

```powershell
python 06_cv.py
```

**실제 정답 실행으로 확인한 출력**

```text
Tree fold F1: [0.6667, 0.5455, 0.6667, 0.5, 0.3333] 평균: 0.5424 표준편차: 0.1236
Forest fold F1: [0.3333, 0.0, 0.6667, 0.4, 0.0] 평균: 0.28 표준편차: 0.2544
XGB fold F1: [0.4, 0.6, 0.6667, 0.3333, 0.5] 평균: 0.5 표준편차: 0.1229
사용 자료: train60 내부만 / valid20·test20 미사용
```

**생성 파일:** `outputs/06_cv_scores.csv`

**완료 기준:** TODO가 실제 계산·모델 호출에 사용되고, 위 출력 및 단위·행 수가 일치하며 다음 질문에 답할 수 있어야 합니다. 반올림 끝자리 차이만 있다면 검증 버전부터 확인합니다.

- fold 평가12건은 보관한 test20건과 같은가요?
- 표준편차를 신뢰구간이라고 불러도 될까요?

## 실습 07~08 · 분류 4후보 통합

**목표:** Dummy·Tree·Forest·XGB의 CV와 valid 결과를 재현합니다.

**열 파일:** `07_project.py`  
**완성할 위치:** `A07a`, `A07b`

- **A07a**: train 내부 교차검증 뒤 전체 train으로 학습하세요.
- **A07b**: valid의 예측을 prediction에 넣으세요.

**힌트:** 전처리·저장·반복 구조는 제공되었습니다.

```powershell
python 07_project.py
```

**실제 정답 실행으로 확인한 출력**

```text
model  CV_F1  CV_std  train_F1  valid_F1  FP  FN
 Dummy 0.0000  0.0000    0.0000    0.0000   0   6
  Tree 0.5424  0.1236    0.7097    0.3333   4   4
Forest 0.2800  0.2544    0.8000    0.2222   2   5
   XGB 0.5000  0.1229    0.7742    0.1818   4   5
CV 평균 F1 최대, 동률이면 미리 제시한 후보 순서로 선택합니다. valid는 별도 진단입니다.
저장: outputs/classification_comparison.csv, cv_scores.csv, validation_predictions.csv, feature_importance.csv
test는 학습·변환·예측·평가하지 않았습니다.
```

**생성 파일:** `outputs/classification_comparison.csv`, `outputs/cv_scores.csv`, `outputs/validation_predictions.csv`, `outputs/feature_importance.csv`

**완료 기준:** TODO가 실제 계산·모델 호출에 사용되고, 위 출력 및 단위·행 수가 일치하며 다음 질문에 답할 수 있어야 합니다. 반올림 끝자리 차이만 있다면 검증 버전부터 확인합니다.

- CV와 valid 순위가 다를 때 무엇을 보고하나요?
- 동률일 때의 선택 규칙을 미리 정했나요?
- 이 표를 최종 test 성능이라고 불러도 되나요?

## 실습 08 · 재실행과 짝 확인

`python 07_project.py`를 다시 실행하고 4개 결과 CSV가 재생성되는지 봅니다. `classification_comparison.csv`의 CV_F1·CV_std는 train60 안의5-fold 결과이며, valid_F1·FP·FN은 바깥 valid20 결과입니다. 서로 다른 평가 자료를 같은 점수로 부르지 않습니다.

Forest 설정은100개·깊이4·최소 잎2, Tree는깊이3·최소 잎3, XGB는80회차·깊이2·학습률0.05입니다. 전체 설정은 코드에 명시되어 있습니다. MDI 표는 **Forest의 train60 학습 결과, 변환8열**이며 모든 후보의 중요도를 합친 표가 아닙니다.

| 설명 항목 | 나의 설명 |
|---|---|
| 주 지표와 양성 라벨 | 여기에 작성 |
| CV 평균 F1 최대 선택 및 동점 규칙 | 후보 목록 순서를 유지하며 설명 |
| valid 결과·FP·FN으로 본 한계 | 여기에 작성 |
| 숫자로 뒷받침한 근거 3개 | 여기에 작성 |
| 중요도 1위가 원인이 아닌 이유 | 여기에 작성 |
| test 보관 여부 / 짝의 재현 결과 | 여기에 작성 |

이 수준에서는 저장 모델·복원 코드를 요구하지 않습니다. 보고서는 이 README의 표로 대체할 수 있습니다. Git 기록은 이미 사용하는 저장소가 있을 때 README·코드·CSV를 커밋하는 선택 활동입니다.
