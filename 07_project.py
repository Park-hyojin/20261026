"""실습 07~08 응용: 한 파일에서 회귀·분류 각각 세 후보를 비교합니다."""
from pathlib import Path
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix

## CSV에서 데이터 읽어오기
# raw = 사용할 데이터를 저장할 변수(DataFrame)
# .set_index() : pandas 기본 인텍스(0,1,2 ...)를 request_id(R001, R002, ...)로 설정.
# 이유 : raw.loc["R001"]처럼 ID를 기준으로 하기 위해.
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")

## 역할별 ID 나누기
# 이 표는 전날의 분할 소속과 각 분할 내부의 행 순서(position)를 함께 보존합니다.
# split = raw 데이터를 나눌 때 사용할 변수(DataFrame)
# - split_manifest.csv : ID("R001", "R002", ...), 역할("train", "valid", "test"), 순서(0, 1, 2, ...)가 저장된 파일

# train_ids = train 역할을 할 request_id들을 담을 변수
# - split.loc[split["split"] == "train"] : train 역할 만 고름
# - .sort_values("position") : position 기준으로 오름차순(0, 1, 2, ...)
# - ["request_id"] : request_id 열만 추출 ["R009", "R026", "R074", ...]
split = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = split.loc[split["split"] == "train"].sort_values("position")["request_id"]
valid_ids = split.loc[split["split"] == "valid"].sort_values("position")["request_id"]

## ID별 raw 분배
# train = raw에서 특정 ID를 뽑아 train 역할을 수행할 데이터
# valid = raw에서 특정 ID를 뽑아 valid 역할을 수행할 데이터
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]

## 모델 예측에 사용할 입력 특성(열) 선택
# features = 사용할 입력 특성(열) 4개
# X_train = train에서 뽑아낸 데이터 변수 (학습용)
# X_valid = valid에서 뽑아낸 데이터 변수 (검증용)
features = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train = train[features]
X_valid = valid[features]

# test의 ID는 분할표에만 남겨 둡니다. test 입력·정답을 모델에 전달하지 않습니다.

## 전처리 후 통합
# 제공: 전날 만든 전처리입니다. 숫자 2열은 중앙값→표준화, 문자 2열은 최빈값→원핫입니다.
# num = 숫자형 데이터 전처리 pipeline
# cat = 범주형 데이터 전처리 pipeline
# pre = 숫자형·범주형 전처리 결합
num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
pre = ColumnTransformer([("num", num, features[:2]), ("cat", cat, features[2:])])

## 저장할 변수 생성
# 제공 K: PPT 실습06의 동일 원본·분할·standard 탐색 결과(분류3, 회귀15)를 재사용합니다.
# regression_rows = 회귀 모델별 평가 지표(MAE, RMSE) 저장 변수
# classification_rows = 분류 모델별 평가 지표(F1, Precision, Recall, FP, FN) 저장 변수
# predictions =  각 모델의 실제값("actual")과 예측값("predicted") 저장
regression_rows = []
classification_rows = []
predictions = []

## 회귀 모델
# Dummy : 학습 데이터의 정답 평균값으로 예측하는 기준 모델 (예측값 - 1235.15, 1235.15, 1235.15)
# Linear : 입력 특성과 정답의 선형 관계를 학습하는 회귀 모델 (예측값 - 1131.480, 1110.693, 1435.497)
# KNN15 : 가장 가까운 학습 데이터 15개의 정답 평균으로 예측하는 모델 (예측값 - 1238.133, 1191.4, 1347.866)
# name = 모델 이름 ("Dummy", "Linear", "KNN15")
# estimator = 사용할 회귀 모델
for name, estimator in [("Dummy", DummyRegressor()), ("Linear", LinearRegression()),
                        ("KNN15", KNeighborsRegressor(n_neighbors=15))]:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])

    # TODO A07r: 회귀 모델을 학습하고 valid 예측을 구하세요.
    # 힌트: 정답 열은 elapsed_ms입니다.

    ## 모델 학습
    # .fit(입력 특성, 정답) : 입력 특성과 정답 사이의 관계를 학습
    # - X_train(입력 특성)은 "input_chars", "allowed_tools", "task_type", "priority"
    # - X_train(입력 특성)에 예측해야 할 정답인 "elapsed_ms"가 있으면 안된다.
    # prediction = 예측값 (numpy.ndarray, X_valid 수 만큼 (20개))
    # .predict(입력 특성) : 입력 특성과 학습된 관계로, 정답을 예측
    model.fit(X_train, train["elapsed_ms"])
    prediction = model.predict(X_valid)

    # 위는 train["elapsed_ms"] (모델 학습에 사용)
    # 아래는 valid["elapsed_ms"] (예측 성능 평가에 사용)
    # 둘 다 정답값이다.

    ## 평가 지표
    # mean_absolute_error(실제값, 예측값) : 평균 절대 오차(MAE) 계산
    # root_mean_squared_error(실제값, 예측값) : 평균 제곱근 오차(RMSE) 계산
    # 오차가 적은 모델이 좋다.
    regression_rows.append({"model": name, "scale": "standard",
                            "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction),
                            "RMSE_ms": root_mean_squared_error(valid["elapsed_ms"], prediction)})
    predictions.append(pd.DataFrame({"request_id": valid.index, 
                                     "task": "regression", "model": name,
                                     "actual": valid["elapsed_ms"].to_numpy(), "predicted": prediction}))

## 분류모델
# Dummy : 학습 데이터에서 가장 빈도가 높은 클래스로 예측하는 기준 모델
# Logistic : 입력 특성과 분류 결과의 관계를 학습하여 각 클래스의 확률을 계산하는 모델
# KNN3 : 가장 가까운 학습 데이터 3개의 다수결로 분류하는 모델
for name, estimator in [("Dummy", DummyClassifier(strategy="most_frequent")),
                        ("Logistic", LogisticRegression(C=1.0, max_iter=1000)),
                        ("KNN3", KNeighborsClassifier(n_neighbors=3))]:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])

    # TODO A07c: 분류 모델을 학습하고 valid 예측을 구하세요.
    # 힌트: 정답 열은 failed입니다.
    ## 모델 학습
    # .fit(입력 특성, 정답) : 입력 특성과 정답 사이의 관계를 학습
    # - X_train(입력 특성)에 예측해야 할 정답인 "failed"가 있으면 안된다.
    # prediction = 예측값 (numpy.ndarray, X_valid 수 만큼 (20개))
    # .predict(입력 특성) : 입력 특성과 학습된 관계로, 정답을 예측
    model.fit(X_train, train["failed"])
    prediction = model.predict(X_valid)

    # 이번에는 분류라 "elapsed_ms" 대신에 "failed"를 사용.

    ## 평가 지표
    # 각종 분류용 평가 지표
    # Precision, Recall, F1 은 높을수록 좋음
    # FP, FN 는 낮을수록 좋음
    tn, fp, fn, tp = confusion_matrix(valid["failed"], prediction, labels=[0, 1]).ravel()
    classification_rows.append({"model": name, "scale": "standard",
                                "F1": f1_score(valid["failed"], prediction, zero_division=0),
                                "Precision": precision_score(valid["failed"], prediction, zero_division=0),
                                "Recall": recall_score(valid["failed"], prediction, zero_division=0), 
                                "FP": fp, 
                                "FN": fn})
    predictions.append(pd.DataFrame({"request_id": valid.index, 
                                     "task": "classification", "model": name,
                                     "actual": valid["failed"].to_numpy(), "predicted": prediction}))

regression = pd.DataFrame(regression_rows)
classification = pd.DataFrame(classification_rows)
out = HERE / "outputs"
out.mkdir(exist_ok=True)

regression.to_csv(out / "regression_comparison.csv", index=False, encoding="utf-8-sig")
classification.to_csv(out / "classification_comparison.csv", index=False, encoding="utf-8-sig")
pd.concat(predictions, ignore_index=True).to_csv(out / "validation_predictions.csv", index=False, encoding="utf-8-sig")

print("회귀 / valid 20건")
print(regression.round(3).to_string(index=False))
print("분류 / valid 20건")
print(classification.round(3).to_string(index=False))
print("저장: outputs의 비교표 2개와 validation_predictions.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")

# 회귀는 또이또이한데, 분류는 KNN3가 좋음.


"""

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

"""