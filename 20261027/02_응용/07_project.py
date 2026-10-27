"""실습 07~08 응용: 얕은 폴더 구조에서 분류 실험표를 완성합니다."""
from pathlib import Path
import pandas as pd

# 제공: 전날의 분할 소속뿐 아니라 position 순서까지 그대로 읽습니다.
## 데이터 로드 및 분할
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
manifest = pd.read_csv(HERE / "data/split_manifest.csv")

train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]

FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train, X_valid = train[FEATURES], valid[FEATURES]
y_train, y_valid = train["failed"], valid["failed"]

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
# test 20건의 입력·정답은 아래 모델에 전달하지 않습니다.

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# 제공: 전날 배운 결측 대체와 문자 변환입니다. 통계는 fit할 때만 배웁니다.
## 전처리기 설정
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
])
# 트리는 이 실습에서 표준화하지 않습니다. 각 후보·fold마다 새 전처리기를 학습합니다.

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix

## 교차 검증 및 예측 모델 설정
# folds : 5-Fold 교차 검증의 분할 설정 및 인덱스 생성.
#         (학습 인덱스, 평가 인덱스) 튜플 5개를 담은 리스트
# candidates : 비교할 예측 모델 4개 설정
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))
candidates = [("Dummy", DummyClassifier(strategy="most_frequent")),
              ("Tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)), 
              ("Forest", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2, 
                                                max_features="sqrt", random_state=42, n_jobs=1)), 
              ("XGB", XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                                    objective="binary:logistic", eval_metric="logloss",
                                    tree_method="hist", device="cpu", random_state=42, n_jobs=1))]

## 결과 저장용 변수 및 DataFrame 생성
# predictions : 검증 데이터의 ID·정답 및 모델별 예측값을 저장하는 DataFrame
rows, cv_rows = [], []
predictions = pd.DataFrame({"request_id": X_valid.index, "actual": y_valid.to_numpy()})

## 모델별 교차 검증 및 예측
for name, estimator in candidates:
    ## Pipeline 생성
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])

    # TODO A07a: train 내부 교차검증 뒤 전체 train으로 학습하세요.
    ## 교차 검증
    scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1")

    # TODO A07b: valid의 예측을 prediction에 넣으세요.
    ## 모델 학습
    model.fit(X_train, y_train)
    ## 검증 데이터(valid) 예측
    prediction = model.predict(X_valid)

    ## 성능 평가
    # confusion_matrix() : 실제 정답과 예측값을 비교하여 TN, FP, FN, TP 계산
    tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()

    # rows : 모델별 교차 검증 및 학습·검증 성능 저장
    rows.append({"model": name, 
                 "CV_F1": scores.mean(), 
                 "CV_std": scores.std(),
                 "train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
                 "valid_F1": f1_score(y_valid, prediction, zero_division=0), 
                 "FP": fp, 
                 "FN": fn})
    
    ## 교차 검증 결과 정리
    # cv_rows : 각 Fold의 번호, 학습·평가 데이터 수, F1 점수 저장
    for number, value in enumerate(scores, start=1):
        cv_rows.append({"model": name, 
                        "fold": number, 
                        "fit_n": 48, 
                        "evaluate_n": 12, 
                        "F1": value})

    ## 예측 결과 정리
    # predictions["모델이름"]에 모델별 예측값을 저장
    predictions[name] = prediction

    ## Forest 특성 중요도 정리 (불순도 기반)
    if name == "Forest":
        importance = pd.DataFrame({"feature": model["pre"].get_feature_names_out(),
                                   "importance": model["model"].feature_importances_})

## 결과 DataFrame 생성 및 CSV 저장
# classification_comparison.csv : 모델별 교차 검증 및 학습·검증 성능 비교
# cv_scores.csv : 모델별 Fold의 학습·평가 데이터 수와 F1 점수
# validation_predictions.csv : 검증 데이터의 ID, 실제 정답 및 모델별 예측값
# feature_importance.csv : RandomForest의 MDI 기반 특성 중요도
comparison = pd.DataFrame(rows)
comparison.to_csv(OUT / "classification_comparison.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(cv_rows).to_csv(OUT / "cv_scores.csv", index=False, encoding="utf-8-sig")
predictions.to_csv(OUT / "validation_predictions.csv", index=False, encoding="utf-8-sig")
importance.to_csv(OUT / "feature_importance.csv", index=False, encoding="utf-8-sig")

## 결과 출력
print(comparison.round(4).to_string(index=False))
print("CV 평균 F1 최대, 동률이면 미리 제시한 후보 순서로 선택합니다. valid는 별도 진단입니다.")
print("저장: outputs/classification_comparison.csv, cv_scores.csv, validation_predictions.csv, feature_importance.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")


"""

 model  CV_F1  CV_std  train_F1  valid_F1  FP  FN
 Dummy 0.0000  0.0000    0.0000    0.0000   0   6
  Tree 0.5424  0.1236    0.7097    0.3333   4   4
Forest 0.2800  0.2544    0.8000    0.2222   2   5
   XGB 0.5333  0.0919    0.7742    0.1818   4   5
CV 평균 F1 최대, 동률이면 미리 제시한 후보 순서로 선택합니다. valid는 별도 진단입니다.
저장: outputs/classification_comparison.csv, cv_scores.csv, validation_predictions.csv, feature_importance.csv
test는 학습·변환·예측·평가하지 않았습니다.

"""