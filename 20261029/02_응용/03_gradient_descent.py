"""실습 03: 매번 새로운 그래프를 만들고 gradient를 비웁니다."""
from pathlib import Path
import csv
import torch

w = torch.tensor(0.0, requires_grad=True)
lr = 0.1
history = []
for step in range(1, 11):
    loss = (w - 4.0) ** 2   # 오차를 계산하는 손실 함수
    loss.backward()         # 손실 함수의 기울기 계산
    with torch.no_grad():

        # TODO A03: 위 요구사항을 읽고 이 블록을 완성하세요.
        w -= lr * w.grad    # 기울기의 반대 방향으로 가중치 갱신

    w.grad.zero_()          # 누적된 기울기를 0으로 초기화

    # 갱신한 w를 기준으로 손실을 다시 계산하여 기록
    # w.detach() : 계산 그래프에서 분리된 Tensor 반환
    history.append([step, w.item(), (w.detach() - 4.0).square().item()])

for step, value, loss_after in history:
    print(step, round(value, 6), round(loss_after, 6))

out = Path(__file__).resolve().parent / "outputs"
out.mkdir(exist_ok=True)
with (out / "gd_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f); writer.writerow(["step", "w_after", "loss_after"]); writer.writerows(history)


"""
A03 : 경사하강법(Gradient Descent)을 이용한 가중치 갱신 및 손실 감소 확인
경사하강법(Gradient Descent) : 현재 위치에서 기울기를 확인하고 손실이 줄어드는 방향으로 조금씩 이동하는 방법
                             손실(Loss)을 최소화하도록 모델의 가중치를 최적화하기 위해 사용

1 0.8 10.240001
2 1.44 6.5536
3 1.952 4.194304
4 2.3616 2.684354
5 2.68928 1.717987
6 2.951424 1.099511
7 3.161139 0.703687
8 3.328911 0.45036
9 3.463129 0.28823
10 3.570503 0.184467

"""