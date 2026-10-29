"""실습 05: 마지막 작은 batch를 버리지 않고 자료를 모두 읽습니다."""
from pathlib import Path
import csv
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

X = torch.arange(20, dtype=torch.float32).reshape(10, 2) / 10
y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
dataset = TensorDataset(X, y)   # TensorDataset : 입력 데이터(X)와 정답(y)을 묶어 데이터셋 생성

# dataset을 batch_size=4, shuffle=False, drop_last=False로 공급하세요.
# 정상 데이터 10건이 4·4·2로 나뉘어야 합니다. num_workers=0을 유지합니다.
# TODO A05: 위 요구사항을 읽고 이 블록을 완성하세요.
# DataLoader : 데이터셋을 배치 단위로 나누어 공급하는 클래스
# - batch_size=4 : 데이터를 최대 4개씩 묶음
# - shuffle=False : 데이터 순서를 섞지 않음
# - drop_last=False : 마지막에 남는 작은 배치도 사용
# - num_workers=0 : 메인 프로세스에서 데이터 로딩
loader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False, num_workers=0)

sizes = []
for index, (xb, yb) in enumerate(loader, 1):
    sizes.append([index, len(yb)])
    print(index, tuple(xb.shape), tuple(yb.shape))

print("합계:", sum(n for _, n in sizes))
logits = torch.tensor([[2., 0.], [0., 2.]])
# nn.CrossEntropyLoss() : 예측 점수(logits)와 정답을 비교하여 분류 손실 계산
print("예시 CE:", round(nn.CrossEntropyLoss()(logits, torch.tensor([0, 1])).item(), 6))

out = Path(__file__).resolve().parent / "outputs"
out.mkdir(exist_ok=True)
with (out / "batch_sizes.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f); writer.writerow(["batch", "samples"]); writer.writerows(sizes)

# 추가 A09: 배치 크기가 다를 때 평균 loss를 합치는 방법입니다.
# 입력: 평균 손실 [0.1, 0.2, 0.9], 각 배치의 표본 수 [4, 4, 2].
# 요구: loss × 실제 표본 수의 합을 전체 표본 수로 나누세요.
# 금지: 배치 평균 세 개를 단순 평균하거나 마지막 배치를 4건으로 계산하지 않습니다.
# 완료: weighted_loss=0.3. 잘못된 단순 평균 0.4와 차이를 설명합니다.
loss_means = [0.1, 0.2, 0.9]
sample_counts = [4, 4, 2]

# TODO A09: 배치 크기까지 반영한 평균을 계산하세요.
# zip() : 배치별 평균 손실과 표본 수를 같은 순서끼리 묶음
weighted_loss = sum(loss * count for loss, count in zip(loss_means, sample_counts)) / sum(sample_counts)

print("샘플 가중 loss:", round(weighted_loss, 6))
print("배치 단순 평균:", round(sum(loss_means) / len(loss_means), 6))


"""
A05 : DataLoader를 이용한 배치 분할 및 전체 데이터 처리 확인

1 (4, 2) (4,)
2 (4, 2) (4,)
3 (2, 2) (2,)
합계: 10
예시 CE: 0.126928

A09 : 배치별 표본 수를 반영한 가중 평균 손실 계산

샘플 가중 loss: 0.3
배치 단순 평균: 0.4

"""