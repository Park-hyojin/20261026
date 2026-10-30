
"""
실습 A06: BatchNorm 적용 여부에 따른 모델 성능 비교

목적: BatchNorm 적용 여부(False, True)에 따른 검증 성능 비교 및 동작 확인

실행 흐름:
1. 데이터 준비
2. 실험 조건 설정
3. 모델 생성
4. 모델 학습 및 평가
5. 결과 분석 및 저장
6. BatchNorm 동작 진단
"""

from pathlib import Path
import copy
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 1. 데이터 준비
torch.set_num_threads(1)
base = Path(__file__).resolve().parent
out = base / "outputs"
out.mkdir(exist_ok=True)
device = torch.device("cpu")

df = pd.read_csv(base / "data" / "points.csv")
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32).to(device)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long).to(device)
loss_fn = nn.CrossEntropyLoss()


# 2. 실험 조건 설정
# bn=False와 bn=True만 다른 두 설정을 구성하세요.
# batch_size=32는 동일하고 마지막 16건도 학습합니다. BatchNorm을 사용하는 train batch는 2건 이상이어야 합니다.
# TODO A06: 위 요구사항에 맞게 이 블록만 완성합니다.
configs = [
    {"name": "baseline", "lr": 0.01, "bn": False},
    {"name": "batchNorm", "lr": 0.01, "bn": True},
]
summary = []

for config in configs:
    # 3. 모델 생성
    # 동일한 초기화 Seed를 사용해 BatchNorm 적용 여부 비교
    torch.manual_seed(42)
    width = config.get("width", 32)
    layers = [nn.Linear(2, width)]
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, width)])
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, 2)])
    model = nn.Sequential(*layers).to(device)

    # 4. 모델 학습 및 평가
    # 동일한 데이터 셔플 순서를 사용하고 마지막 불완전 배치도 학습
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(43), num_workers=0)

    if config.get("optimizer", "AdamW") == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],
                                    momentum=config.get("momentum", 0.0), weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.0)

    history = []
    for epoch in range(1, 41):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        # 평가 모드에서 Train Loss, Valid Loss 및 Valid Accuracy 측정
        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(X_train.to(device)), y_train.to(device)).item()
            logits = model(X_valid)
            valid_loss = loss_fn(logits, y_valid).item()
            valid_acc = (logits.argmax(1) == y_valid).float().mean().item()
        history.append([epoch, train_loss, valid_loss, valid_acc])

    # 5. 결과 분석 및 저장
    # 최저 Valid Loss를 기준으로 최적 Epoch 선정
    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss", "valid_acc"])
    best = table.loc[table["valid_loss"].idxmin()]
    name = config["name"]
    table.to_csv(out / f"06_{name}_history.csv", index=False)
    summary.append({**config, "best_epoch": int(best["epoch"]),
                    "best_valid_loss": float(best["valid_loss"]),
                    "valid_acc_at_best": float(best["valid_acc"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss})

    # Train/Valid Loss 변화와 최적 Epoch를 그래프로 저장
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"06_{name}_curves.png", dpi=150)
    plt.close()
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6))

# BatchNorm 적용 여부에 따른 결과를 하나의 비교 CSV로 저장
result = pd.DataFrame(summary)
result.to_csv(out / "06_comparison.csv", index=False)
print("생성: outputs/06_*.csv, *.png / test 미사용")


# 6. BatchNorm 동작 진단
# 마지막으로 학습한 BatchNorm 모델을 복사해 원본의 상태 변경 방지
probe = copy.deepcopy(model)
xb = X_train[:32].to(device)

# 동일한 입력에서 학습 모드와 평가 모드의 출력 차이 비교
# train 모드의 BatchNorm은 누적 통계 갱신
probe.train()
with torch.no_grad():
    train_output = probe(xb)

probe.eval()
with torch.no_grad():
    eval_output = probe(xb)

print("진단용 복사 모델 train/eval 최대 차이:", round((train_output - eval_output).abs().max().item(), 6))
