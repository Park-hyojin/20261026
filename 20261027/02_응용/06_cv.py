"""실습 06 응용: Pipeline 전체를 같은 train 5-fold로 평가합니다."""
from pathlib import Path
import pandas as pd

# 제공: 전날의 분할 소속뿐 아니라 position 순서까지 그대로 읽습니다.
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
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
])
# 트리는 이 실습에서 표준화하지 않습니다. 각 후보·fold마다 새 전처리기를 학습합니다.

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))
candidates = [("Tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)), 
              ("Forest", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
                                                max_features="sqrt", random_state=42, n_jobs=1)), 
              ("XGB", XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                                    objective="binary:logistic", eval_metric="logloss",
                                    tree_method="hist", device="cpu", random_state=42, n_jobs=1))]
rows = []

for name, estimator in candidates:
    ## 모델 생성
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])

    # TODO A06: 전처리까지 포함한 model의 fold별 F1을 계산하세요.
    ## 교차 검증(Cross-Validation) 실행
    # cross_val_score() : 각 그룹을 번갈아 평가 데이터로 사용하여 모델 성능 측정
    # model, X_train, y_train : 평가할 모델, 학습용 입력 데이터, 정답
    # cv=folds : 검증 방법=정의한 교차 검증
    # scoring="f1" : 점수 기준=F1
    scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1")

    print(name, "fold F1:", scores.round(4).tolist(), "평균:", round(scores.mean(), 4), "표준편차:", round(scores.std(), 4))

    for number, value in enumerate(scores, start=1):
        rows.append({"model": name, "fold": number, "fit_n": 48, "evaluate_n": 12, "F1": value})

pd.DataFrame(rows).to_csv(OUT / "06_cv_scores.csv", index=False, encoding="utf-8-sig")
print("사용 자료: train60 내부만 / valid20·test20 미사용")
