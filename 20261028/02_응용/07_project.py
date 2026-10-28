"""실습 07~08 응용: 06번 계획을 읽어 개발80으로 새 Pipeline을 학습하고 test20을 평가합니다."""
### 1. 라이브러리 불러오기
## 1-1. 파일 처리와 데이터 분석 도구
from pathlib import Path
import json
import pandas as pd
import numpy as np
## 1-2. 전처리와 모델 학습·평가 도구
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

### 2. 확정 계획과 데이터 준비
## 2-1. 확정 계획과 원본 데이터 불러오기
# 06번에서 확정한 모델 설정·임계값·데이터 분할 정보 불러오기
# 원본 데이터의 요청 ID를 인덱스로 지정해 분할 정보와 연결
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
plan = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
## 2-2. 개발·테스트 데이터 분리와 입력 특성 설정
# 확정된 분할 ID에 따라 개발 80%와 테스트 20% 데이터 구성
# 확정된 입력 특성을 추출하고 수치형·범주형으로 구분
ids = plan["split_ids"]
develop = raw.loc[ids["train"] + ids["valid"]]
test = raw.loc[ids["test"]]
X_develop, X_test = develop[plan["features"]].copy(), test[plan["features"]].copy()
NUM, CAT = plan["features"][:2], plan["features"][2:]
## 2-3. 계획에 따른 파생 특성 생성
# 파생 특성 사용 시 개발·테스트에 동일한 공식 적용
# 결측값 대체·표준화 통계는 개발 데이터 학습 과정에서 계산
# 파생 특성 = 기존 열을 가공해 만든 새로운 입력 열
if plan["derived"]:
    for part in (X_develop, X_test):
        part["log_input_chars"] = np.log1p(part["input_chars"])
        part["chars_per_slot"] = part["input_chars"] / (1.0 + part["allowed_tools"])
    NUM = NUM + ["log_input_chars", "chars_per_slot"]

### 3. 전처리와 모델 구성
## 3-1. 수치형·범주형 전처리 구성
# 수치형은 결측값을 중앙값으로 채운 뒤 표준화(StandardScaler)
# 범주형은 결측값을 최빈값으로 채운 뒤 원핫 인코딩(OneHotEncoder)
# 학습 시 없던 범주는 해당 특성의 원핫 인코딩 결과를 모두 0으로 처리
pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), 
                      ("scaler", StandardScaler())]), NUM),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
])
## 3-2. 전처리와 로지스틱 회귀를 Pipeline으로 연결
# 전처리와 분류 모델을 하나의 흐름으로 연결
# 06번에서 확정한 규제 매개변수(C)와 최대 반복 횟수 적용
model = Pipeline([("pre", pre), ("model", LogisticRegression(C=plan["C"], max_iter=plan["max_iter"]))])

### 4. 확정 모델 학습과 테스트 예측
## 4-1. 개발 데이터로 새 Pipeline 학습
# 개발 데이터 80%로 전처리와 모델을 새로 학습
# 입력 특성과 실제 실패 여부(failed)를 이용해 지도학습 수행
model.fit(X_develop, develop["failed"])
## 4-2. 테스트 데이터의 실패 확률 예측
# 테스트에는 학습된 전처리와 모델만 적용
# 클래스 순서를 확인해 실패(1)에 해당하는 확률만 선택
positive_column = model.classes_.tolist().index(1)
probability = model.predict_proba(X_test)[:, positive_column]
## 4-3. 확정 임계값으로 최종 예측 생성
# 실패 확률이 임계값보다 크면 실패(1), 이하면 성공(0)으로 분류
prediction = (probability > plan["threshold"]).astype(int)

### 5. 테스트 성능 평가
## 5-1. 혼동행렬 계산
# 실제 실패 여부와 예측을 비교해 TN·FP·FN·TP 건수 계산
# TN: 성공을 성공으로, TP: 실패를 실패로 예측한 건수
# FP: 성공을 실패로, FN: 실패를 성공으로 예측한 건수
tn, fp, fn, tp = confusion_matrix(test["failed"], prediction, labels=[0, 1]).ravel()
## 5-2. 평가 지표 정리
# 정확도(Accuracy)·정밀도(Precision)·재현율(Recall)·F1은 최종 예측값으로 계산
# ROC AUC는 실패 확률을 이용해 성공·실패의 순위 구분 능력 평가
# 테스트 건수·실제 실패 건수·혼동행렬의 각 항목을 함께 정리
metrics = {"n": len(test), 
           "Accuracy": accuracy_score(test["failed"], prediction),
           "Precision": precision_score(test["failed"], prediction, zero_division=0),
           "Recall": recall_score(test["failed"], prediction, zero_division=0),
           "F1": f1_score(test["failed"], prediction, zero_division=0),
           "ROC_AUC": roc_auc_score(test["failed"], probability),
           "TN": int(tn), 
           "FP": int(fp), 
           "FN": int(fn), 
           "TP": int(tp), 
           "actual_failed": int(test["failed"].sum())}

### 6. 결과 저장과 출력
## 6-1. 예측 결과표 생성
# 요청별 ID·실제 실패 여부·실패 확률·최종 예측을 하나의 표로 구성
result = pd.DataFrame({"request_id": test.index, "actual": test["failed"].to_numpy(),
                       "probability_failed": probability, "predicted": prediction})
## 6-2. 예측 결과와 평가 지표 저장
# 요청별 예측 결과는 CSV, 전체 평가 지표는 JSON으로 저장
result.to_csv(OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
(OUT / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
## 6-3. 평가 결과와 오류 사례 출력
# 확정 모델과 평가 지표를 출력하고 오분류 사례를 최대 3건 확인
# 테스트 결과로 모델을 재선택하지 않음
# 08번에서 동일 설정과 원본 데이터로 재실행
print("확정 모델:", plan["selected"], "/ 학습80·최종20")
print(pd.DataFrame([metrics]).round(4).to_string(index=False))
print("오류 사례")
print(result.loc[result["actual"] != result["predicted"]].head(3).round(3).to_string(index=False))
print("저장: outputs/test_predictions.csv, test_metrics.json")
print("08번에서는 이 설정·원본을 그대로 재실행합니다. test를 보고 후보를 다시 고르지 않습니다.")


"""

 n  Accuracy  Precision  Recall     F1  ROC_AUC  TN  FP  FN  TP  actual_failed
20      0.75        0.6     0.5 0.5455   0.7857  12   2   3   3              6
오류 사례
request_id  actual  probability_failed  predicted
      R060       1               0.100          0
      R017       1               0.379          0
      R081       0               0.590          1
저장: outputs/test_predictions.csv, test_metrics.json
08번에서는 이 설정·원본을 그대로 재실행합니다. test를 보고 후보를 다시 고르지 않습니다.

"""
