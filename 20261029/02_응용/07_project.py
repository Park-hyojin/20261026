"""실습 07~08: 앞 실습을 한 파일로 연결하고 환경·학습·평가 결과를 저장합니다."""

### 1. 라이브러리 불러오기
from pathlib import Path
import csv
import json
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

### 2. 데이터 준비
## 2-1. 경로 설정과 데이터 불러오기
# 현재 파일을 기준으로 경로 설정
# CSV 데이터를 딕셔너리 리스트로 저장
base = Path(__file__).resolve().parent
out = base / "outputs"
out.mkdir(exist_ok=True)
with (base / "data" / "points.csv").open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

## 2-2. 학습·검증 데이터 분리와 텐서 변환
# split 열을 기준으로 학습(train)·검증(valid) 데이터 분리
# x1·x2를 실수형 입력 텐서로 변환
# label을 정수형 정답 텐서로 변환
train_rows = [r for r in rows if r["split"] == "train"]
valid_rows = [r for r in rows if r["split"] == "valid"]
X_train = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in train_rows])
y_train = torch.tensor([int(r["label"]) for r in train_rows], dtype=torch.long)
X_valid = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in valid_rows])
y_valid = torch.tensor([int(r["label"]) for r in valid_rows], dtype=torch.long)
# test 40건은 모델에 전달하지 않습니다. 순서를 바꾸거나 다시 분할하지 않습니다.

### 3. 학습 환경과 모델 구성
## 3-1. 난수 시드·실행 장치·DataLoader 설정
# 모델 초기화에 사용할 난수 시드 설정
# CUDA 사용이 가능하면 GPU, 아니면 CPU 선택
# DataLoader : 학습 데이터를 섞어 32개씩 불러오며 셔플 시드 43 설정
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                    generator=torch.Generator().manual_seed(43), num_workers=0)

## 3-2. 모델·손실 함수·옵티마이저 구성
# 입력 2개 → 은닉 뉴런 8개(ReLU) → 출력 2개로 구성하고 실행 장치로 이동
# CrossEntropyLoss : 분류 손실 함수
# SGD : 가중치 갱신을 위한 옵티마이저(학습률 0.1)
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

## 3-3. 평가용 데이터와 기록 준비
# 학습·검증 텐서를 실행 장치로 이동
# history : 학습·검증 손실과 검증 정확도 기록
Xv, yv = X_valid.to(device), y_valid.to(device)
Xt, yt = X_train.to(device), y_train.to(device)
history = []

### 4. 모델 학습과 평가
for epoch in range(81):
    ## 4-1. 미니배치 학습
    # epoch=1부터 80까지 학습 모드로 미니배치 학습 수행
    # 입력과 정답을 실행 장치로 이동하고 누적된 기울기 초기화
    # 모델의 출력과 정답을 비교해 손실 계산
    if epoch > 0:
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            # A07 : 손실을 역전파해 기울기를 계산하고 가중치 갱신
            loss.backward()
            optimizer.step()

    ## 4-2. 학습·검증 성능 측정과 기록
    # 학습 전(epoch=0)과 이후 10에포크마다 모델 성능 측정
    # 평가 모드로 전환하고 기울기 계산 없이 학습·검증 손실 계산
    # logits : 검증 데이터의 클래스별 점수
    # 가장 높은 점수의 클래스를 정답과 비교해 검증 정확도 계산
    # 에포크·학습 손실·검증 손실·검증 정확도를 기록하고 출력
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(Xt), yt).item()
            logits = model(Xv)
            valid_loss = loss_fn(logits, yv).item()
            accuracy = (logits.argmax(dim=1) == yv).float().mean().item()
        history.append([epoch, train_loss, valid_loss, accuracy])
        print(epoch, round(train_loss, 6), round(valid_loss, 6), round(accuracy, 4))

### 5. 결과 저장과 오답 분석
## 5-1. 학습 기록 저장
# 에포크·학습 손실·검증 손실·검증 정확도를 CSV로 저장
with (out / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss_eval", "valid_loss", "valid_accuracy"])
    writer.writerows(history)

## 5-2. 최종 예측과 평가 지표 저장
# 평가 모드에서 기울기 계산 없이 검증 데이터의 클래스별 점수 계산
# 최종 학습·검증 성능과 실행 정보를 final_metrics.json으로 저장
model.eval()
with torch.no_grad():
    logits = model(Xv)
    # A08 : 클래스별 점수가 가장 큰 인덱스를 예측값으로 선택하고 CPU로 이동
    predictions = logits.argmax(dim=1).cpu()

metrics = {"device": str(device), "torch": str(torch.__version__), "epochs": 80,
           "train_rows": len(train_rows), "valid_rows": len(valid_rows), "test_used": False,
           "initial_train_loss": history[0][1], "final_train_loss": history[-1][1],
           "valid_loss": history[-1][2], "valid_accuracy": history[-1][3]}
(out / "final_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

## 5-3. 예측 결과와 실행 환경 저장
# validation_predictions.csv : 검증 데이터의 ID·실제값·예측값 저장
# gpu_check.txt : PyTorch·CUDA 정보와 선택된 실행 장치 저장
with (out / "validation_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "actual", "predicted"])
    writer.writerows((r["id"], int(y), int(p)) for r, y, p in zip(valid_rows, y_valid, predictions))

(out / "gpu_check.txt").write_text(
    f"torch={torch.__version__}\ncuda_build={torch.version.cuda}\n"
    f"cuda_available={torch.cuda.is_available()}\nselected_device={device}\n", encoding="utf-8")
print("생성: outputs의 환경·학습·검증 결과")

## 5-4. 오답 수집과 저장
# A10 : 실제값과 예측값이 다른 검증 데이터만 기존 순서대로 수집
# 오답의 ID·실제값·예측값을 CSV로 저장하고 오답 수 출력
errors = [
    (r["id"], int(y), int(p))
    for r, y, p in zip(valid_rows, y_valid, predictions)
    if int(y) != int(p)
]
with (out / "validation_errors.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "actual", "predicted"])
    writer.writerows(errors)
print("valid 오답 수:", len(errors))


"""
실행 결과

A07: 역전파 후 옵티마이저로 가중치 갱신
A08: 클래스별 최대 점수의 인덱스를 구하고 CPU로 이동
A10: 실제 오답만 원래 순서대로 수집

0 0.565832 0.510637 0.7
10 0.331469 0.281476 0.975
20 0.228078 0.182579 0.975
30 0.176405 0.134341 1.0
40 0.145632 0.105641 1.0
50 0.12504 0.086818 1.0
60 0.110228 0.074174 1.0
70 0.098804 0.064484 1.0
80 0.089844 0.057217 1.0
생성: outputs의 환경·학습·검증 결과
valid 오답 수: 0

"""
