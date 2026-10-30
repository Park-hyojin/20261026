
"""
실습 A07~A10: Optimizer 및 Dropout 설정에 따른 모델 성능 비교

목적: SGD·AdamW 및 Dropout 설정에 따른 검증 성능 비교와 최적 모델 평가

실행 흐름:
1. 데이터 준비
2. 실험 조건 설정
3. 모델 생성
4. 모델 학습 및 평가
5. 최적 모델 선정 및 복원
6. 클래스별 성능 평가
7. 결과 분석 및 저장
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
device = torch.device("cuda:0")

df = pd.read_csv(base / "data" / "points.csv")
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32).to(device)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long).to(device)
loss_fn = nn.CrossEntropyLoss()


# 2. 실험 조건 설정
# Optimizer, 학습률 및 Dropout 비율을 변경한 네 가지 조건 비교
configs = [{"name": "sgd_lr0.1", "optimizer": "SGD", "lr": 0.1, "dropout": 0.0},
           {"name": "sgd_lr0.05", "optimizer": "SGD", "lr": 0.05, "dropout": 0.0},
           {"name": "adamw", "optimizer": "AdamW", "lr": 0.02, "dropout": 0.0},
           {"name": "adamw_dropout", "optimizer": "AdamW", "lr": 0.02, "dropout": 0.3},]
summary = []

for config in configs:
    # 3. 모델 생성
    # 동일한 초기화 Seed를 사용하고 설정에 따라 Dropout 적용
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
    # 동일한 데이터 셔플 순서와 Batch Size로 학습
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(43), num_workers=0)

    if config.get("optimizer", "AdamW") == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],
                                    momentum=config.get("momentum", 0.0), weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.0)

    history = []
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, 151):
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

            # Valid Loss가 개선되면 최적 Epoch와 독립된 가중치 복사본 저장
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state = copy.deepcopy(model.state_dict())

        history.append([epoch, train_loss, valid_loss, valid_acc])

    # 5. 최적 모델 선정 및 복원
    # 최저 Valid Loss를 기준으로 최적 Epoch 선정
    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss", "valid_acc"])
    best = table.loc[table["valid_loss"].idxmin()]
    name = config["name"]

    summary.append({**config, "best_epoch": int(best["epoch"]),
                    "best_valid_loss": float(best["valid_loss"]),
                    "valid_acc_at_best": float(best["valid_acc"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss})

    # 학습 중 저장한 최적 가중치를 복원하고 Valid Loss 재확인
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_pred = model(X_valid).argmax(1).cpu()
        restored_loss = loss_fn(model(X_valid), y_valid).item()

    summary[-1]["restored_valid_loss"] = restored_loss
    summary[-1]["restored_matches_best"] = abs(restored_loss - best_loss) < 1e-6

    # 6. 클래스별 성능 평가
    # 실제 정답을 기준으로 클래스별 Support, Correct 및 Recall 계산
    labels = y_valid.cpu()
    class_rows = []

    for label in (0, 1):
        mask = labels == label
        support = mask.sum().item()
        correct = (best_pred[mask] == label).sum().item()
        recall = correct / support if support > 0 else None

        class_rows.append({"label": label,
                           "support": support,
                           "correct": correct,
                           "recall": recall,})

    # 7. 결과 분석 및 저장
    # Epoch별 학습 기록과 클래스별 평가 및 예측 결과 저장
    table.to_csv(out / f"07_{name}_history.csv", index=False)
    pd.DataFrame(class_rows).to_csv(out / f"07_{name}_class_metrics.csv", index=False)
    pd.DataFrame({"id": valid["id"].tolist(), 
                  "actual": labels.tolist(),
                  "predicted": best_pred.tolist()}).to_csv(out / f"07_{name}_predictions.csv", index=False)

    # Train/Valid Loss 변화와 최적 Epoch를 그래프로 저장
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"07_{name}_curves.png", dpi=150)
    plt.close()

    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6))

# 실험 조건별 최적 성능과 복원 검증 결과를 하나의 비교 CSV로 저장
result = pd.DataFrame(summary)
result.to_csv(out / "07_comparison.csv", index=False)
print("생성: outputs/07_*.csv, *.png / test 미사용")
