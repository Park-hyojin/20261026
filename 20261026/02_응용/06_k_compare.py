"""실습 06 응용: 분류의 train F1과 valid F1을 기록합니다."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# 이 표는 전날의 분할 소속과 각 분할 내부의 행 순서(position)를 함께 보존합니다.
split = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = split.loc[split["split"] == "train"].sort_values("position")["request_id"]
valid_ids = split.loc[split["split"] == "valid"].sort_values("position")["request_id"]
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]
features = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train = train[features]
X_valid = valid[features]
# test의 ID는 분할표에만 남겨 둡니다. test 입력·정답을 모델에 전달하지 않습니다.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

# 제공: 전날 만든 전처리입니다. 숫자 2열은 중앙값→표준화, 문자 2열은 최빈값→원핫입니다.
num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
pre = ColumnTransformer([("num", num, features[:2]), ("cat", cat, features[2:])])

rows = []
for k in [1, 3, 5, 9, 15]:
    # TODO A06: n_neighbors를 반복 변수 k로 바꾸세요.
    # clone(pre) : 기존 전처리 설정을 복제
    model = Pipeline([("pre", clone(pre)), ("model", KNeighborsClassifier(n_neighbors=k))])

    model.fit(X_train, train["failed"])
    rows.append({"K": k,
        "train_F1": f1_score(train["failed"], model.predict(X_train), zero_division=0),
        "valid_F1": f1_score(valid["failed"], model.predict(X_valid), zero_division=0)})
    
comparison = pd.DataFrame(rows)
# 동점이면 작은 K: valid_F1 내림차순, K 오름차순으로 정렬합니다.
best = comparison.sort_values(["valid_F1", "K"], ascending=[False, True]).iloc[0]
out = HERE / "outputs"
out.mkdir(exist_ok=True)

comparison.to_csv(out / "knn_search.csv", index=False, encoding="utf-8-sig")
print(comparison.round(3).to_string(index=False))
print("선택 K:", int(best["K"]))
print("저장: outputs/knn_search.csv")


"""

 K  train_F1  valid_F1
 1     1.000     0.400
 3     0.571     0.545
 5     0.462     0.200
 9     0.519     0.000
15     0.364     0.000
선택 K: 3
저장: outputs/knn_search.csv

"""