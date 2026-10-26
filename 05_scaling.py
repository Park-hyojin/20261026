"""실습 05 응용: 숫자 경로만 바꾸어 KNN5를 비교합니다."""
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

rows = []
for name, scaler in [("none", "passthrough"), ("standard", StandardScaler()), ("minmax", MinMaxScaler())]:
    # passthrough : 스케일링 생략 (결측값 처리는 유지)
    # 숫자 : 결측값을 중앙값으로 대체 후 스케일링 [생략, 표준화, min-max정규화]
    # 범주 : 결측값을 최빈값으로 대체 후 원핫 인코딩
    # 숫자와 범주형 전처리 결과 결합
    # KNN 분류 모델 적용 (이웃 5개)
    num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", scaler)])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer([("num", num, features[:2]), ("cat", cat, features[2:])])
    model = Pipeline([("pre", pre), ("model", KNeighborsClassifier(n_neighbors=5))])

    # TODO A05: train에서 학습하고 valid 예측을 구하세요.
    # fit(학습데이터, 정답) : 모델 학습
    # predict(검증데이터) : 학습 모델로 검증 데이터 예측
    # prediction : 예측값
    model.fit(X_train, train["failed"])
    prediction = model.predict(X_valid)

    # name : "none", "standard", "minmax"
    # scaler : 스케일링 없음, 표준화, Min-Max 정규화
    # K : 이웃 수 5 (n_neighbors=5)
    # valid_F1 : 검증 데이터의 F1 점수
    # f1_score(정답값, 예측값, ) : F1 점수 계산
    # zero_division=0 : 0으로 나누는 경우 해당 점수를 0으로 처리
    rows.append({"scale": name, "K": 5, "valid_F1": f1_score(valid["failed"], prediction, zero_division=0)})

comparison = pd.DataFrame(rows)
out = HERE / "outputs"
out.mkdir(exist_ok=True)

comparison.to_csv(out / "scaling_comparison.csv", index=False, encoding="utf-8-sig")
print(comparison.round(3).to_string(index=False))
print("저장: outputs/scaling_comparison.csv")


"""

   scale  K  valid_F1
    none  5       0.4
standard  5       0.2
  minmax  5       0.4
저장: outputs/scaling_comparison.csv


"""