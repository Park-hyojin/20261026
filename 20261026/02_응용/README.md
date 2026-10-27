# 응용 · 코드와 입력·결과를 가볍게 나눕니다

기본필수를 마친 수강생의 선택 활동입니다. 직접 정의하는 함수 없이 각 파일에서 계산합니다. 코드7개는 이 폴더에 두고 입력은 `data/`, 생성 결과는 `outputs/`에 둡니다. 전체 TODO8곳입니다.

```text
02_응용/
  README.md
  01_linear.py ... 07_project.py
  data/requests.csv
  data/split_manifest.csv
  outputs/                  # 실행 전에는 비어 있음
```

`outputs/`가 비어 있는 것은 누락이 아닙니다. 파일을 실행하면 생성합니다. 다른 실습에서 만든 결과가 없어도 다음 실습을 시작할 수 있습니다. 07은 06의 제공 K 선택값(분류3·회귀15)을 재사용하며, 이를 새로 계산했다고 주장하지 않습니다.

VS Code에서 이 폴더를 열고 전날의 Python 환경을 선택합니다. 현재 터미널 위치도 이 폴더로 둡니다. 설치 안내는 [상위 README](../README.md)를 봅니다.

## 데이터와 고정 조건

기존 PPT의 요청100건과 분할 ID·행 순서를 유지합니다. `split_manifest.csv`의 request_id는 요청ID, split은 train/valid/test, position은 **해당 분할 내부의 순서**입니다. train60·valid20·test20이며 파일에서 다시 무작위로 나누지 않습니다.

모델 입력4열은 `input_chars`, `allowed_tools`, `task_type`, `priority`입니다. `elapsed_ms`는 회귀 정답, `failed`는 분류 정답입니다. `status`, `retry_count`, 실제 실행시간은 실행 후 정보이므로 입력에 넣지 않습니다. ID도 입력에서 제외합니다.

숫자의 중앙값 대체·표준화와 문자의 최빈값 대체·원핫 변환은 제공 코드입니다. 모두 train에서 학습합니다. 직접 예외처리나 데이터 무결성 검사 도구를 작성하는 과제가 아닙니다.

## 01 · 입력 두 개의 선형회귀

PPT 14쪽과 연결 · 권장 실습 20분

**목표:** 각 입력과 계수를 연결한다.

**열 파일:** `01_linear.py`. TODO만 먼저 완성합니다. 제공된 읽기·저장 코드를 다시 작성하지 않아도 됩니다.

**새 함수·옵션:** coef_[0]는 글자 수 계수, coef_[1]는 도구 수 계수이다.

**직접 완성할 부분**

- **A01**: 두 입력과 정답으로 model을 학습하세요.
  힌트: X는 이미 4행×2열입니다.

**실행:** 현재 README가 있는 폴더에서 아래 명령을 사용합니다.

```powershell
python 01_linear.py
```

**예상 출력** — 소수점 출력 자릿수·공백 표시는 환경에 따라 약간 다를 수 있습니다.

```text
계수: [  0.5 100. ]
절편: 500.0
[400, 2] 예측(ms): [900.]
손계산(ms): 900.0
```

**완료 기준:** 계수 [0.5,100], 절편 500, 예측 900ms를 확인한다.

**설명 질문:** 입력 열의 순서를 바꾸면 같은 뜻인가?


---

## 02 · 기준선과 선형회귀의 오차 비교

PPT 23쪽과 연결 · 권장 실습 25분

**목표:** 기준선과 모델을 같은 자료·지표로 비교한다.

**열 파일:** `02_regression.py`. TODO만 먼저 완성합니다. 제공된 읽기·저장 코드를 다시 작성하지 않아도 됩니다.

**새 함수·옵션:** DummyRegressor(): train 평균만 예측 / clone(pre): 전처리 독립 / 두 지표 단위는 ms

**직접 완성할 부분**

- **A02**: 회귀 정답으로 학습하고 같은 valid를 예측하세요.
  힌트: fit의 정답은 train["elapsed_ms"]입니다.

**실행:** 현재 README가 있는 폴더에서 아래 명령을 사용합니다.

```powershell
python 02_regression.py
```

**예상 출력** — 소수점 출력 자릿수·공백 표시는 환경에 따라 약간 다를 수 있습니다.

```text
 model  MAE_ms  RMSE_ms
 Dummy 541.260  658.745
Linear 551.797  639.996
저장: outputs/regression_comparison.csv
```

**완료 기준:** 회귀 비교표 2행을 저장하고 각 지표에서 낮은 모델을 읽는다.

**설명 질문:** MAE와 RMSE가 서로 다른 모델을 고를 수 있는가?

**생성되는 파일:** `outputs/regression_comparison.csv`


---

## 03 · 실패 확률과 혼동행렬

PPT 34쪽과 연결 · 권장 실습 20분

**목표:** 정답이 있는 valid에서 오탐과 놓침을 구분한다.

**열 파일:** `03_logistic.py`. TODO만 먼저 완성합니다. 제공된 읽기·저장 코드를 다시 작성하지 않아도 됩니다.

**새 함수·옵션:** confusion_matrix(labels=[0,1]): [[TN,FP],[FN,TP]] / C=1: 고정 규제 설정 / max_iter=1000: 반복 상한

**직접 완성할 부분**

- **A03**: 실패 여부로 학습하고 X_valid를 예측하세요.
  힌트: 실패1의 정답 열은 failed입니다.

**실행:** 현재 README가 있는 폴더에서 아래 명령을 사용합니다.

```powershell
python 03_logistic.py
```

**예상 출력** — 소수점 출력 자릿수·공백 표시는 환경에 따라 약간 다를 수 있습니다.

```text
클래스 순서: [0 1]
앞 4건 실패 확률: [0.141 0.185 0.601 0.745]
혼동행렬 / 행=실제, 열=예측, 순서=[0,1]
[[10  4]
 [ 4  2]]
valid F1: 0.333
```

**완료 기준:** TN·FP·FN·TP=10·4·4·2를 해석한다.

**설명 질문:** 확률이 큰 요청이라고 항상 실제 실패인가?


---

## 04 · 거리 가중 평균과 단순 평균

PPT 42쪽과 연결 · 권장 실습 25분

**목표:** 이웃 수와 자료를 고정하고 가중 방식의 차이를 이해한다.

**열 파일:** `04_knn.py`. TODO만 먼저 완성합니다. 제공된 읽기·저장 코드를 다시 작성하지 않아도 됩니다.

**새 함수·옵션:** weights="uniform": 단순 평균 / weights="distance": 거리 역수 가중 평균

**직접 완성할 부분**

- **A04**: weights에 반복 변수 weight를 넣으세요.
  힌트: 이웃 수는 3으로 고정합니다.

**실행:** 현재 README가 있는 폴더에서 아래 명령을 사용합니다.

```powershell
python 04_knn.py
```

**예상 출력** — 소수점 출력 자릿수·공백 표시는 환경에 따라 약간 다를 수 있습니다.

```text
uniform 예측: [23.33]
distance 예측: [28.06]
```

**완료 기준:** 23.33과 28.06을 대조한다.

**설명 질문:** 값40에 큰 비중이 부여되는 이유는 무엇인가?


---

## 05 · K를 고정하고 스케일링 비교

PPT 52쪽과 연결 · 권장 실습 25분

**목표:** 한 번에 한 조건만 바꾸는 비교를 수행한다.

**열 파일:** `05_scaling.py`. TODO만 먼저 완성합니다. 제공된 읽기·저장 코드를 다시 작성하지 않아도 됩니다.

**새 함수·옵션:** none: 원단위 / standard: 평균·표준편차 / minmax: train 최소·최대 / passthrough: 해당 단계를 통과

**직접 완성할 부분**

- **A05**: train에서 학습하고 valid 예측을 구하세요.
  힌트: 앞 실습과 학습·예측 흐름이 같습니다.

**실행:** 현재 README가 있는 폴더에서 아래 명령을 사용합니다.

```powershell
python 05_scaling.py
```

**예상 출력** — 소수점 출력 자릿수·공백 표시는 환경에 따라 약간 다를 수 있습니다.

```text
   scale  K  valid_F1
    none  5       0.4
standard  5       0.2
  minmax  5       0.4
저장: outputs/scaling_comparison.csv
```

**완료 기준:** none 0.4·standard 0.2·minmax 0.4를 확인한다.

**설명 질문:** standard의 점수가 낮으니 코드 오류인가?

**생성되는 파일:** `outputs/scaling_comparison.csv`


---

## 06 · 분류 K 후보 다섯 개 비교

PPT 61쪽과 연결 · 권장 실습 25분

**목표:** 후보·고정 조건·선택 기준을 기록한다.

**열 파일:** `06_k_compare.py`. TODO만 먼저 완성합니다. 제공된 읽기·저장 코드를 다시 작성하지 않아도 됩니다.

**새 함수·옵션:** sort_values: 사전 규칙으로 정렬 / idx와 순위를 자의적으로 바꾸지 않기

**직접 완성할 부분**

- **A06**: n_neighbors를 반복 변수 k로 바꾸세요.
  힌트: K만 바꾸고 전처리와 분할은 고정합니다.

**실행:** 현재 README가 있는 폴더에서 아래 명령을 사용합니다.

```powershell
python 06_k_compare.py
```

**예상 출력** — 소수점 출력 자릿수·공백 표시는 환경에 따라 약간 다를 수 있습니다.

```text
 K  train_F1  valid_F1
 1     1.000     0.400
 3     0.571     0.545
 5     0.462     0.200
 9     0.519     0.000
15     0.364     0.000
선택 K: 3
저장: outputs/knn_search.csv
```

**완료 기준:** 분류 K=3을 선택하고 K=1의 train F1=1과 valid F1을 비교한다.

**설명 질문:** 선택에 사용한 valid F1을 최종 성능이라고 부를 수 있는가?

**생성되는 파일:** `outputs/knn_search.csv`


---

## 07 · 회귀·분류 각각 세 후보 비교

PPT 73쪽과 연결 · 권장 실습 60분 (07·08 합계)

**목표:** 이전 실습의 모델과 지표를 한 실행으로 연결한다.

**열 파일:** `07_project.py`. TODO만 먼저 완성합니다. 제공된 읽기·저장 코드를 다시 작성하지 않아도 됩니다.

**새 함수·옵션:** 회귀 주 지표 RMSE 최소 / 분류 주 지표 F1 최대 / 동일 문제 안에서만 비교

**직접 완성할 부분**

- **A07r**: 회귀 모델을 학습하고 valid 예측을 구하세요.
  힌트: 정답 열은 elapsed_ms입니다.
- **A07c**: 분류 모델을 학습하고 valid 예측을 구하세요.
  힌트: 정답 열은 failed입니다.

**실행:** 현재 README가 있는 폴더에서 아래 명령을 사용합니다.

```powershell
python 07_project.py
```

**예상 출력** — 소수점 출력 자릿수·공백 표시는 환경에 따라 약간 다를 수 있습니다.

```text
회귀 / valid 20건
 model    scale  MAE_ms  RMSE_ms
 Dummy standard 541.260  658.745
Linear standard 551.797  639.996
 KNN15 standard 530.063  641.788
분류 / valid 20건
   model    scale    F1  Precision  Recall  FP  FN
   Dummy standard 0.000      0.000   0.000   0   6
Logistic standard 0.333      0.333   0.333   4   4
    KNN3 standard 0.545      0.600   0.500   2   3
저장: outputs의 비교표 2개와 validation_predictions.csv
test는 학습·변환·예측·평가하지 않았습니다.
```

**완료 기준:** 각 비교표 3행과 valid 예측 120행(20건×3모델×2문제)을 저장한다.

**설명 질문:** Linear와 KNN의 회귀 RMSE 차이가 작을 때 무엇을 함께 말해야 하는가?

**생성되는 파일:** `outputs/regression_comparison.csv`, `outputs/classification_comparison.csv`, `outputs/validation_predictions.csv`


## 08교시 · 같은 파일로 재실행하고 설명 (35분)

07번 파일을 저장하고 한 번 더 실행합니다. 기존 CSV가 덮어써져 같은 수치를 내는지 확인합니다. 짝에게 아래 네 질문을 말로 설명하고, 이 README 아래 빈칸에 각 한 문장씩 작성합니다. 자동 채점 프로그램이나 추가 Python 파일은 만들지 않습니다.

1. 회귀의 정답·주 지표·선택 모델: ______
2. 분류의 정답·양성 라벨·주 지표·선택 모델(동점 여부 포함): ______
3. 예측 CSV에서 한 요청 ID를 골라 실제값과 예측값을 비교: ______
4. test를 쓰지 않은 이유와 이번 결과의 한계: ______

점수가 같을 때의 간단한 규칙: 이 실습은 먼저 등록한 기준선처럼 **더 단순한 후보를 유지**합니다. valid를 바꾸거나 오류가 큰 행을 삭제하여 원하는 순위를 만들지 않습니다.

Git은 전날 배운 내용을 유지하는 선택 기록 활동입니다. 새 브랜치·새 협업 도구를 추가로 배우지 않습니다. 로컬 저장소가 이미 있으면 수정한 코드·README·CSV의 변경만 확인하고 커밋합니다. Git이 처음이면 강사 도움을 받아 수행하며, 기본 개념 설명을 대체하지 않습니다.

## 막혔을 때 확인

`NoneType` 또는 학습 전 오류가 나오면 먼저 해당 파일의 TODO가 남아 있는지 확인합니다. `NotFittedError`는 fit 부분이 아직 `pass`인지 봅니다. 모델 설정 TODO는 파일이 실행되더라도 잘못된 결과를 낼 수 있습니다.

`ModuleNotFoundError`가 나면 VS Code가 전날의 가상환경을 선택했는지 확인합니다. `root_mean_squared_error`를 찾지 못하면 설치된 scikit-learn 버전을 확인하고 역할 폴더의 `requirements.txt`를 사용합니다.

`파일을 찾을 수 없음`이면 ZIP을 완전히 해제했는지, 코드와 함께 배포된 데이터가 이동·삭제되지 않았는지 확인합니다. 출력 CSV를 Excel에서 열어 둔 경우 저장이 막힐 수 있으므로 닫은 뒤 다시 실행합니다.

오류 메시지를 가리는 `try/except`를 추가하는 과제가 아닙니다. TODO와 파일 위치를 확인하고 계속 막히면 실행 명령과 오류 마지막 줄을 강사에게 보여 줍니다.

## 공식 참고자료

아래 문서는 함수 동작을 더 확인할 때 사용하는 참고 자료이며, 링크를 모두 읽어야 실습을 할 수 있는 것은 아닙니다. 예제 출력은 README에 적은 설치 버전에서 실행해 확인했습니다.

- [선형회귀](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LinearRegression.html)
- [로지스틱회귀](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [최근접 이웃](https://scikit-learn.org/1.8/modules/neighbors.html)
- [표준화](https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [RMSE](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.root_mean_squared_error.html)
- [데이터 누수와 Pipeline](https://scikit-learn.org/stable/common_pitfalls.html)
- [모델 저장·복원 주의점](https://scikit-learn.org/stable/model_persistence.html)
