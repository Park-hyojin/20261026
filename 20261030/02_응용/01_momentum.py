
"""
실습 A01: SGD Momentum에 따른 모델 성능 비교

목적: Momentum(0.0, 0.9)에 따른 검증 성능 비교

실행 흐름:
1. 데이터 준비
2. 실험 조건 설정
3. 모델 생성
4. 모델 학습 및 평가
5. 결과 분석 및 저장
"""

from pathlib import Path
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
# 두 설정은 momentum만 0.0과 0.9로 다르게 만드세요.
# optimizer=SGD, lr=0.05, 나머지 설정은 동일하게 유지합니다.
# TODO A01: 위 요구사항에 맞게 이 블록만 완성합니다.
configs = [
    {"name": "momentum0", "optimizer": "SGD", "lr": 0.05, "momentum": 0.0},
    {"name": "momentum9", "optimizer": "SGD", "lr": 0.05, "momentum": 0.9}
]
summary = []

for config in configs:
    # 3. 모델 생성
    # 초기 가중치와 모델 구조를 동일하게 설정해 Momentum만 비교
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
    # 각 설정에서 동일한 데이터 셔플 순서를 사용하도록 Seed 고정
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(43), num_workers=0)

    if config.get("optimizer", "AdamW") == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],
                                    momentum=config.get("momentum", 0.0), weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.0)

    history = []
    for epoch in range(1, 21):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        # 학습 후 평가 모드에서 Train/Valid Loss와 정확도 측정
        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(X_train.to(device)), y_train.to(device)).item()
            logits = model(X_valid)
            valid_loss = loss_fn(logits, y_valid).item()
            valid_acc = (logits.argmax(1) == y_valid).float().mean().item()
        history.append([epoch, train_loss, valid_loss, valid_acc])

    # 5. 결과 분석 및 저장
    # Valid Loss가 가장 낮은 Epoch를 최적 결과로 선정
    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss", "valid_acc"])
    best = table.loc[table["valid_loss"].idxmin()]
    name = config["name"]
    table.to_csv(out / f"01_{name}_history.csv", index=False)
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
    plt.savefig(out / f"01_{name}_curves.png", dpi=150)
    plt.close()
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6))

# Momentum별 결과를 하나의 비교 CSV로 저장
result = pd.DataFrame(summary)
result.to_csv(out / "01_comparison.csv", index=False)
print("생성: outputs/01_*.csv, *.png / test 미사용")
