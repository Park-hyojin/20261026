# Day10 · 응용

Optimizer, 학습률, 배치 크기, 모델 폭, Dropout, BatchNorm을 바꾸며 학습 과정과 검증 성능을 비교하는 실습입니다.
개별 실험에서 확인한 내용을 바탕으로, 마지막에는 여러 학습 설정을 비교하고 최적 모델을 복원해 클래스별 성능까지 평가합니다.

**학습 목표**

- 비교할 조건을 구분하고, 각 설정이 학습에 미치는 영향을 설명합니다.
- 학습·검증 손실 곡선을 읽고 수렴 양상과 과적합 여부를 판단합니다.
- 학습·평가 모드를 올바르게 전환하고, 검증 손실을 기준으로 최적 모델을 복원해 평가합니다.

### [01_momentum.py](01_momentum.py) · Momentum 비교

SGD의 학습률을 0.05로 고정하고 momentum 0.0과 0.9를 비교합니다.
이전 기울기를 반영하는 Momentum이 학습 속도와 검증 손실에 어떤 영향을 주는지 살펴봅니다.

실행 결과:

```text
momentum0 best epoch: 20 best valid loss: 0.332401
momentum9 best epoch: 19 best valid loss: 0.069346
생성: outputs/01_*.csv, *.png / test 미사용
```

### [02_learning_rate.py](02_learning_rate.py) · 학습률 비교

AdamW의 학습률을 0.001, 0.01, 0.1로 바꾸어 학습합니다.
학습곡선에서 손실이 줄어드는 속도와 출렁임, 최저 검증 손실을 비교합니다.

실행 결과:

```text
lr0.001 best epoch: 20 best valid loss: 0.293574
lr0.01 best epoch: 17 best valid loss: 0.038566
lr0.1 best epoch: 16 best valid loss: 0.026041
생성: outputs/02_*.csv, *.png / test 미사용
```

### [03_batches.py](03_batches.py) · 배치 크기 비교

학습 데이터 240건을 배치 크기 8과 32로 나누어 반복 횟수와 마지막 배치 크기를 확인합니다.
모델 학습 없이 DataLoader의 동작을 살펴보며, 나누어떨어지지 않는 마지막 배치도 유지합니다.

실행 결과:

```text
batch: 8 iterations: 30 마지막: 8 처리 수: 240
batch: 32 iterations: 8 마지막: 16 처리 수: 240
```

### [04_width.py](04_width.py) · 모델 폭 비교

은닉층의 뉴런 수를 8과 64로 바꾸어 각각 40 epoch 동안 학습합니다.
모델 크기에 따른 학습·검증 손실의 변화를 비교하고, 두 곡선의 차이로 과적합 여부를 살펴봅니다.

실행 결과:

```text
width8 best epoch: 40 best valid loss: 0.099038
width64 best epoch: 40 best valid loss: 0.007299
생성: outputs/04_*.csv, *.png / test 미사용
```

### [05_dropout.py](05_dropout.py) · Dropout 비교

Dropout 확률 0.0과 0.3을 비교해 검증 성능의 변화를 확인합니다.
학습할 때는 일부 뉴런의 출력을 무작위로 0으로 만들고, 평가할 때는 `model.eval()`로 Dropout을 끕니다.

실행 결과:

```text
dropout0.0 best epoch: 38 best valid loss: 0.01656
dropout0.3 best epoch: 40 best valid loss: 0.033904
생성: outputs/05_*.csv, *.png / test 미사용
```

### [06_batchnorm.py](06_batchnorm.py) · BatchNorm 비교

BatchNorm을 사용하지 않은 모델과 사용한 모델의 학습곡선을 비교합니다.
학습 모드에서는 현재 배치의 통계를, 평가 모드에서는 누적 통계를 사용하며, 같은 입력의 train·eval 출력 차이도 확인합니다.

실행 결과:

```text
baseline best epoch: 38 best valid loss: 0.01656
batchNorm best epoch: 21 best valid loss: 0.032923
생성: outputs/06_*.csv, *.png / test 미사용
진단용 복사 모델 train/eval 최대 차이: 1.919039
```

### [07_project.py](07_project.py) · 통합 실험과 최적 모델 평가

SGD 두 설정(lr 0.1, 0.05)과 AdamW 두 설정(lr 0.02, Dropout 0.0, 0.3)을 각각 150 epoch 동안 학습합니다.
검증 손실이 가장 낮은 시점의 가중치를 복사·복원하고, 클래스별 Recall과 예측 결과를 저장합니다.
현재 실행 장치는 `cuda:0`이며, CPU에서 실행하려면 `device`를 `torch.device("cpu")`로 변경합니다.

실행 결과:

```text
sgd_lr0.1 best epoch: 145 best valid loss: 0.039315
sgd_lr0.05 best epoch: 148 best valid loss: 0.077482
adamw best epoch: 107 best valid loss: 0.003392
adamw_dropout best epoch: 125 best valid loss: 0.004879
생성: outputs/07_*.csv, *.png / test 미사용
```



