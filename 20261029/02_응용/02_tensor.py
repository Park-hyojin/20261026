"""실습 02: reshape와 broadcasting을 확인합니다."""
import torch

a = torch.arange(6, dtype=torch.float32)
# 원소 6개를 2행 3열로 바꾸어 matrix에 저장하세요.
# 아래 bias는 각 열에 더해집니다.

# TODO A02: 위 요구사항을 읽고 이 블록을 완성하세요.
# .reshape(2, 3) : Tensor를 2행 3열로 형태 변환(reshape)
matrix = a.reshape(2, 3)
bias = torch.tensor([10., 20., 30.])

# Broadcasting : 크기가 다른 Tensor의 형태를 연산 규칙에 따라 자동으로 맞추는 기능
result = matrix + bias

print("변환 전:", tuple(a.shape))
print("행렬:", tuple(matrix.shape), matrix.tolist())
print("보정 후:", tuple(result.shape), result.tolist())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
moved = result.to(device)
print("이동 장치:", moved.device)
print("CPU에서 확인:", moved.cpu().tolist())


"""
A02 : Tensor의 형태 변환(reshape) 및 Broadcasting 연산 확인

변환 전: (6,)
행렬: (2, 3) [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
보정 후: (2, 3) [[10.0, 21.0, 32.0], [13.0, 24.0, 35.0]]
이동 장치: cuda:0
CPU에서 확인: [[10.0, 21.0, 32.0], [13.0, 24.0, 35.0]]

"""