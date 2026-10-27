"""실습 02 응용: 기존 PPT의 5개 깊이 후보를 비교합니다."""
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
## 전처리기 (ColumnTransformer)
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
])
# 트리는 이 실습에서 표준화하지 않습니다. 각 후보·fold마다 새 전처리기를 학습합니다.

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

rows = []
for depth in [1, 2, 3, 5, None]:
    # TODO A02: depth를 실제 트리 깊이로 적용하세요.
    estimator = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=1, random_state=42)

    ## Pipeline 생성
    # "pre" : 전처리 단계의 이름,       clone(pre) : 전처리기 복제
    # "model" : 예측 모델 단계의 이름,  estimator : 예측 모델 DecisionTreeClassifier
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])

    ## 학습
    model.fit(X_train, y_train)
    
    ## 예측 및 성능 평가
    rows.append({"depth": str(depth), 
                 "train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
                 "valid_F1": f1_score(y_valid, model.predict(X_valid), zero_division=0),
                 "leaves": model["model"].get_n_leaves()})
    
result = pd.DataFrame(rows)
result.to_csv(OUT / "02_depth.csv", index=False, encoding="utf-8-sig")
print(result.round(3).to_string(index=False))


"""

depth  train_F1  valid_F1  leaves
    1     0.500     0.286       2
    2     0.500     0.286       3
    3     0.710     0.333       4
    5     0.882     0.500       9
 None     1.000     0.500      14

"""