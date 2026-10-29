"""실습 04: 은닉 너비에 따라 파라미터 수가 변합니다. 학습 성능 비교는 아닙니다."""
import torch
from torch import nn

torch.manual_seed(42)
X = torch.ones(4, 2)

# hidden : 은닉층의 뉴런 개수 (4, 8, 16)
for hidden in [4, 8, 16]:
    # hidden 값을 사용해 입력 2개 → 은닉 hidden개 → 출력 2개를 연결하세요.
    # 은닉층 뒤에 ReLU를 넣고 마지막에 softmax는 넣지 않습니다.
    # TODO A04: 위 요구사항을 읽고 이 블록을 완성하세요.

    # nn.Linear(2, hidden) : 입력 특성 2개를 은닉층 뉴런 hidden개로 변환
    # nn.ReLU() : 은닉층에 비선형 활성화 함수 적용
    # nn.Linear(hidden, 2) : 은닉층 출력을 클래스 점수 2개로 변환
    model = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    logits = model(X)
    count = sum(p.numel() for p in model.parameters())      # p.numel() : 각 파라미터 Tensor의 원소 개수 반환
    print("은닉:", hidden, "출력:", tuple(logits.shape), "파라미터:", count)

"""
A04 : 은닉층 뉴런 수에 따른 신경망 파라미터 수 변화 확인

은닉: 4 출력: (4, 2) 파라미터: 22
은닉: 8 출력: (4, 2) 파라미터: 42
은닉: 16 출력: (4, 2) 파라미터: 82

"""