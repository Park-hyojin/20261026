"""실습 01 응용: 잎 안 정답의 평균으로 숫자를 예측합니다."""
from sklearn.tree import DecisionTreeRegressor

X = [[100], [200], [300], [700], [800], [900]]
y = [100, 120, 140, 700, 700, 740]
model = DecisionTreeRegressor(max_depth=1, random_state=42)

# TODO A01: 회귀 트리를 학습하고 세 입력을 예측하세요.
model.fit(X, y)
prediction = model.predict([[250], [850], [1200]])

# 회귀 트리 모델은 깊이에 따라 기준을 나누고
# 입력값이 속한 리프 노드의 정답(y) 평균을 예측값으로 준다.
print("분기 기준:", model.tree_.threshold[0])
print("250·850·1200 예측:", prediction.round(2).tolist())
print("왼쪽 평균:", sum(y[:3]) / 3, "오른쪽 평균:", sum(y[3:]) / 3)
print("1200은 학습 범위 밖입니다. 트리도 기존 잎 값으로 예측합니다.")


"""

분기 기준: 500.0
250·850·1200 예측: [120.0, 720.0, 720.0]
왼쪽 평균: 120.0 오른쪽 평균: 720.0
1200은 학습 범위 밖입니다. 트리도 기존 잎 값으로 예측합니다.

"""