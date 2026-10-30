
"""
실습 A03: Batch Size에 따른 데이터 처리 비교

목적: Batch Size(8, 32)에 따른 반복 횟수와 마지막 배치 크기 비교

실행 흐름:
1. 데이터 준비
2. 실험 조건 설정
3. 배치별 데이터 처리
4. 결과 분석 및 저장
"""

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


# 1. 데이터 준비
torch.set_num_threads(1)
base = Path(__file__).resolve().parent
out = base / "outputs"
out.mkdir(exist_ok=True)

df = pd.read_csv(base / "data" / "points.csv")
train = df[df["split"] == "train"]
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)


# 2. 실험 조건 설정
# batch_size 후보를 [8, 32]로 두세요. 총 train 행은 240건입니다.
# drop_last=False이므로 32인 경우 마지막 배치는 16건입니다.
# TODO A03: 위 요구사항에 맞게 이 블록만 완성합니다.
batch_sizes = [8, 32]


# 3. 배치별 데이터 처리
rows = []
for size in batch_sizes:
    # 모든 학습 데이터를 순서대로 처리하고 마지막 불완전 배치도 유지
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=size, shuffle=False)
    counts = [len(yb) for xb, yb in loader]
    rows.append([size, len(counts), counts[-1], sum(counts)])
    print("batch:", size, "iterations:", len(counts), "마지막:", counts[-1], "처리 수:", sum(counts))


# 4. 결과 분석 및 저장
# 배치 크기별 반복 횟수, 마지막 배치 크기, 전체 처리 건수를 CSV로 기록
pd.DataFrame(rows, columns=["batch_size", "iterations", "last_batch", "samples"]).to_csv(
    out / "03_batches.csv", index=False)
