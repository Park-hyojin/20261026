"""실습 06: 80 epoch 학습하고 10 epoch마다 같은 시점의 지표를 기록합니다."""

from pathlib import Path
import csv
import json
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

base = Path(__file__).resolve().parent
out = base / "outputs"
out.mkdir(exist_ok=True)
with (base / "data" / "points.csv").open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
train_rows = [r for r in rows if r["split"] == "train"]
valid_rows = [r for r in rows if r["split"] == "valid"]
X_train = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in train_rows])
y_train = torch.tensor([int(r["label"]) for r in train_rows], dtype=torch.long)
X_valid = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in valid_rows])
y_valid = torch.tensor([int(r["label"]) for r in valid_rows], dtype=torch.long)
# test 40건은 모델에 전달하지 않습니다. 순서를 바꾸거나 다시 분할하지 않습니다.

# DataLoader : 학습 데이터를 배치 단위로 공급
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                    generator=torch.Generator().manual_seed(43), num_workers=0)
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
Xv, yv = X_valid.to(device), y_valid.to(device)
Xt, yt = X_train.to(device), y_train.to(device)
history = []

for epoch in range(81):                             # epoch : 전체 학습 데이터를 한 번 학습하는 단위
    if epoch > 0:
        model.train()                               # model.train() : 모델을 학습 모드로 설정
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()                   # optimizer.zero_grad() : 이전 기울기 초기화
            loss = loss_fn(model(xb), yb)           # 순전파로 손실 계산
            # 현재 batch의 loss로 역전파하고 optimizer를 한 번 갱신하세요.
            # zero_grad와 순전파·손실 계산은 위에 제공되어 있습니다.
            # TODO A06: 위 요구사항을 읽고 이 블록을 완성하세요.
            loss.backward()                         # loss.backward() : 역전파로 기울기 계산
            optimizer.step()                        # SGD를 이용해 모델 파라미터 갱신

    # epoch=0은 학습 전입니다. 로그는 갱신이 끝난 동일 모델로 측정합니다.
    if epoch % 10 == 0:                             # epoch % 10 == 0 : 10 epoch마다 학습 및 검증 지표 기록
        model.eval()                                # model.eval() : 모델을 평가 모드로 설정
        with torch.no_grad():                       # torch.no_grad() : 평가 시 기울기 추적 비활성화
            train_loss = loss_fn(model(Xt), yt).item()      # train_loss : 학습 데이터의 손실
            logits = model(Xv)
            valid_loss = loss_fn(logits, yv).item()         # valid_loss : 검증 데이터의 손실
            accuracy = (logits.argmax(dim=1) == yv).float().mean().item()   # accuracy : 검증 데이터의 정확도
        history.append([epoch, train_loss, valid_loss, accuracy])
        print(epoch, round(train_loss, 6), round(valid_loss, 6), round(accuracy, 4))

with (out / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss_eval", "valid_loss", "valid_accuracy"])
    writer.writerows(history)


"""
A06 : SGD를 이용한 신경망 학습 및 Epoch별 학습·검증 성능 변화 확인

GD와 SGD의 차이
- GD : 전체 학습 데이터를 사용해 기울기를 계산하고 가중치를 갱신
- SGD : 개별 샘플을 사용해 기울기를 계산하고 가중치를 갱신

0 0.565832 0.510637 0.7
10 0.331469 0.281476 0.975
20 0.228078 0.182579 0.975
30 0.176405 0.134341 1.0
40 0.145632 0.105641 1.0
50 0.12504 0.086818 1.0
60 0.110228 0.074174 1.0
70 0.098804 0.064484 1.0
80 0.089844 0.057217 1.0

"""