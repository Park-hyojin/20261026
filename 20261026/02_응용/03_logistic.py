"""실습 03 응용: PPT와 같은 100건에서 실패1의 예측을 확인합니다."""
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

# 제공: 전날 만든 전처리입니다. 숫자 2열은 중앙값→표준화, 문자 2열은 최빈값→원핫입니다.
num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
pre = ColumnTransformer([("num", num, features[:2]), ("cat", cat, features[2:])])

model = Pipeline([("pre", pre), ("model", LogisticRegression(C=1.0, max_iter=1000))])

# TODO A03: 실패 여부로 학습하고 X_valid를 예측하세요.
# 힌트: 실패1의 정답 열은 failed입니다.
model.fit(X_train, train["failed"])
prediction = model.predict(X_valid)

## 출력
# model.classes_ : 정답의 범주와 순서
# model.predict_proba() : 입력 데이터의 클래스별 예측 확률
print("클래스 순서:", model.classes_)
print("앞 4건 실패 확률:", model.predict_proba(X_valid)[:4, 1].round(3))
# 혼돈행렬
# confusion_matrix(실제 정답, 예측 값, 라벨)
print("혼동행렬 / 행=실제, 열=예측, 순서=[0,1]")
print(confusion_matrix(valid["failed"], prediction, labels=[0, 1]))
print("valid F1:", round(f1_score(valid["failed"], prediction, zero_division=0), 3))



"""

클래스 순서: [0 1]
앞 4건 실패 확률: [0.141 0.185 0.601 0.745]
혼동행렬 / 행=실제, 열=예측, 순서=[0,1]
[[10  4]
 [ 4  2]]
valid F1: 0.333

"""