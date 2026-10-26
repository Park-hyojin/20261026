"""실습 01 응용: PPT E02의 계수와 수식을 확인합니다."""
from sklearn.linear_model import LinearRegression

# 열 순서: [글자 수, 허용 도구 수]. y는 실행시간(ms)입니다.
X = [[200, 1], [400, 1], [200, 3], [600, 2]]
y = [700, 800, 900, 1000]
model = LinearRegression()

# TODO A01: 두 입력과 정답으로 model을 학습하세요.
# 힌트: X는 이미 4행×2열입니다.
model.fit(X, y)

# 회귀계수 (기울기)
print("계수:", model.coef_.round(1))
# 절편
print("절편:", round(model.intercept_, 1))
# 학습한 모델로 [400, 2]를 결과 예측
print("[400, 2] 예측(ms):", model.predict([[400, 2]]).round(1))
# 회귀식 손계산
print("손계산(ms):", 500 + 0.5 * 400 + 100 * 2)
