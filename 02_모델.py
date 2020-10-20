# 훈련 데이터에서 패턴을 모델링해서 예측을 수행하고 학습하는 방법
# 사인 함수(sin) -> 사인파
# 사인파 : 사인함수의 결과를 시간에 따라 기록하여 얻는 그래프입니다.

# 종속성
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

# 0 ~ 2까지 임의의 x값을 새성한다.
# 데이터 샘플을 생성합니다.
SAMPLES = 1000

# 시드값을 적용해서 실행할 때마다 다른 랜덤값을 얻게 합니다.
SEED = 1000
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 사인파 진폭의 범위 0 ~ 2n 내에서 균일하게 난수를 생성
x_values = np.random.uniform(low = 0, high = 2*math.pi, size = SAMPLES)

# 값을 섞어서 순서를 따르지 않게 한다.
np.random.shuffle(x_values)

# 사인의 값을 계산합니다.
y_values = np.sin(x_values)
y_values = 0.1 * np.random.randn(*y_values.shape)

# 데이터를 그래프로 그린다.
plt.plot(x_values, y_values, 'h')
plt.show()

# 훈련 60%, 검증 20%, 테스트 20%
TRAIN_SPLIT = int(0.6*SAMPLES)
TEST_SPLIT = int(0.2*SAMPLES + TRAIN_SPLIT)

# np.split을 이용해서 세 부분을 자른다.
# np.split 두번째인수는 데이터가 분할되는 인ㄴ덱스
# 2개의 인덱스를 제동하기 대문에 3개로 나눈다.
x_train, x_validate, x_test = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_validate, y_test = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

# 분할한 데이터를 합쳤을 때 원래 사이즈인지 확인
assert(x_train.size + x_validate.size + x_test.size)

# 분할된 데이터를 각각 다른 색으로 표시
plt.plot(x_train, y_train, 'b.', label = 'Train')
plt.plot(x_validate, y_validate, 'y.', label = 'Validate')
plt.plot(x_test, y_test, 'r.', label = 'Test')
plt.legend()
plt.show()

# 기본 모델 만들기
# 입력값을 사용해서 숫자 출력값(사인 x) 예측하는 모델을 만듭니다.
# 이런 형태의 문제를 회귀문제
# 숫자 출력이 필요한 대부분의 작업에 회귀 모델을 사용하면 매우 간단하게 표현할 수 있다.

# 케라스
import tensorflow as tf
from tensorflow.keras import layers
model_1 = tf.keras.Sequential();

# 뉴런 16 -> 스칼라 값, 활성화함수 Relu 전달
model_1.add(layers.Dense(16, activation='relu', input_shape = (1,))) # layer.Dense : 완전 연결 레이어
# activation은 활성화 함수
# activation = activation_functions(input * weight) + bias
# relu : 입력값이 0보다 크면 입력값을, 입력값이 0보다 작으면 0을 반환
# def relu(input):
    # max(0,0, input)
    
# 활성화 함수를 사용하지 않으면 선형 함수




# 마지막 레이어 뉴런이 하나! 결과값은 1개
# 단일 레이어 -> 기준의 16개 입력을 받은 레이어의 모든 활성화 값을 단일 출력으로 결합한다.
# 이 레이어는 출력 레이어기 때문에 활성화 함수를 지정하지 않고 결과값을 그대로 출력
# 활성화 함수가 없는 대신 다중 입력을 받고 있기 때문에, 각각 입력에 해당하는 가중치가 있습니다.
model_1.add(layers.Dense(1))

# out = sum((inputs * weights)) + bias
# 이 과정에서 훈련 중에 네트워크가 결과를 학습한다.

# 표준 옵티마이저의 손실을 사용해서 회귀 모델을 컴파일
model_1.compile(optimizer='rmsprop', loss = 'mse', metrics = ['mae'])
# optimizer : 훈련중에 네트워크가 입력을 모델링하도록 조정하는 알고리즘을 지정한다.
# 손실 인수 = 훈련과정에서 네트워크 예측이 실제값에서 얼마나 떨어져 있는지 계산하기 위해서 지정한다.
# 손실 함수라고 합니다.
# mse(Mean Squared Error) : 평균제곱오차법
# 숫자를 예측하는 회기문제에서 주로 사용한다.
# metrics : 모델의 성능을 판단하는데 사용되는 함수
# 회기 모형의 성능을 측정하는 'mae'를 주로 사용한다.
# mae(Mean Absolute Error) : 평균절대오차법
# 훈련 중 측정한 결과를 훈련 완료 후 확인할 수 있다.

# 모델 설계를 출력
model_1.summary()
# 모델을 컴파일 한 뒤, 아키텍쳐의 요약정보를 출력
