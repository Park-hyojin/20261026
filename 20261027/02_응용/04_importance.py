"""실습 04 응용: 전처리 출력 이름 8개와 중요도를 대응시킵니다."""
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

from sklearn.ensemble import RandomForestClassifier

model = Pipeline([("pre", clone(pre)), 
                  ("model", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2, 
                                                   max_features="sqrt", random_state=42, n_jobs=1))])

model.fit(X_train, y_train)

# TODO A04: 전처리의 출력 이름과 Forest의 중요도를 가져오세요.
## 특성 정보 추출 (메서드, 속성)
# get_feature_names_out() : 전처리 후 특성 이름 반환
# feature_importances_ : 학습된 모델의 특성 중요도
names = model["pre"].get_feature_names_out()
values = model["model"].feature_importances_

result = pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False)
result.to_csv(OUT / "04_importance.csv", index=False, encoding="utf-8-sig")
print("변환된 특성 수:", len(names))
print(result.round(4).to_string(index=False))
print("중요도는 원인이나 신규 데이터 성능을 보장하지 않습니다.")
