# 07 project 주석

## 코드 구조

1. **데이터 불러오기**
   - `requests.csv` 로드
   - `split_manifest.csv` 로드

2. **데이터 분할**
   - Train / Valid 데이터 구분
   - 입력 특성(X) 선택

3. **데이터 전처리**
   - 숫자형: 결측값 대체 → 표준화
   - 범주형: 결측값 대체 → 원핫 인코딩
   - `ColumnTransformer`로 결합

4. **회귀 모델**
   - Dummy / Linear / KNN15
   - 모델 학습 → 예측 → MAE·RMSE 평가

5. **분류 모델**
   - Dummy / Logistic / KNN3
   - 모델 학습 → 예측 → F1·Precision·Recall·FP·FN 평가

6. **결과 저장 및 출력**
   - 회귀·분류 평가 결과 DataFrame 생성
   - CSV 파일 3개 저장
   - 평가 결과 출력
