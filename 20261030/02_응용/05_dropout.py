
"""
실습 A05: Dropout 비율에 따른 모델 성능 비교

목적: Dropout 비율(0.0, 0.3)에 따른 검증 성능 비교

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
# Dropout 확률 p=0.0과 p=0.3을 비교하세요.
# 평가에서는 model.eval()과 no_grad를 모두 사용하고, best epoch는 valid loss로 찾습니다.
# TODO A05: 위 요구사항에 맞게 이 블록만 완성합니다.
configs = [
    {"name": "dropout0.0", "lr": 0.01, "dropout": 0.0},
    {"name": "dropout0.3", "lr": 0.01, "dropout": 0.3},
]
summary = []

for config in configs:
    # 3. 모델 생성
    # 동일한 초기화 Seed를 사용하고 Dropout 비율만 변경
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
    # 동일한 데이터 셔플 순서를 적용해 Dropout 비율의 영향을 비교
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

        # 평가 모드에서 Dropout을 비활성화하고 Train/Valid 성능 측정
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
    table.to_csv(out / f"05_{name}_history.csv", index=False)
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
    plt.savefig(out / f"05_{name}_curves.png", dpi=150)
    plt.close()
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6))

# Dropout 비율별 결과를 하나의 비교 CSV로 저장
result = pd.DataFrame(summary)
result.to_csv(out / "05_comparison.csv", index=False)
print("생성: outputs/05_*.csv, *.png / test 미사용")
