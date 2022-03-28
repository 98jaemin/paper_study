> # Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (2015)
> ### Sergey Ioffe, Christian Szegedy
</br>

## 1. Introduction
딥러닝은 vision과 speech 등 다양한 분야에서 급속도로 발전하고 있다. 확률적 경사하강법(SGD)는 심층신경망을 학습시키는 효율적인 방식임이 증명되었고, momentum이나 Adagrad와 같은 변형들도 사용되고 있다. SGD의 각 학습 단계에서 크기가 m인 mini-batch를 고려한다. mini-batch는 parameter에 대한 loss function의 gradient를 근사하기 위해 사용된다.

<p align="center"><img src="https://user-images.githubusercontent.com/86872735/159493642-794bb5cc-39e9-42b8-9234-7c076a42425c.png" width="15%"></p>

한번에 하나의 데이터를 사용하는 것과 비교하여 mini-batch를 사용하는 것은 몇 가지 이점이 있다.
- mini-batch loss의 gradient는 전체 학습 데이터의 gradient의 추정값이며 이는 batch size가 커질수록 개선된다
- 각 데이터에 대해 m번 연산하는 것보다 batch를 통해 연산하는 것이 훨씬 효율적이다. (parallelism)

각 layer의 입력은 모든 이전 layer의 parameter에 영향을 받게 되며, 따라서 신경망이 깊어질수록 작은 변화도 증폭되게 된다.

layer의 입력의 분포의 변화는 layer가 계속해서 새로운 분포에 적응해야 한다는 문제를 야기한다. 신경망이 다음과 같은 연산을 한다고 하자. F_1과 F_2는 임의의 변환이고 Theta_1과 Theta_2는 loss를 최소화하기 위해 학습되는 parameter이다.
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/159498055-8b797b0f-7fb9-4983-be10-ecddcb602b11.png" width="22%"></p>

Theta_2에 대한 학습은 x=F_1(u, Theta_1)일 때 다음과 같다.
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/159498647-3822f745-41c3-438f-a8be-896bc46b4f76.png" width="15%"></p>

gradient descent 단계에서 연산을 할 때, 이는 신경망 F_2에 입력 x를 받아들이는 것과 동등하다. 입력 분포의 특성은 sub-network에도 적용된다. x의 분포가 고정된다면 Theta_2는 매번 변화에 맞추어 재조정될 필요가 없다.

입력의 고정된 분포는 sub-network 바깥에도 그 이점이 있다. sigmoid 활성화 함수를 생각해보자. 0 근처가 아닌 모든 x에 대해 기울기 소실 문제가 발생하고 학습이 느리게 이루어진다. 이러한 포화 문제를 해결하기 위해 일반적으로 초기화를 조정하거나 ReLU, 또는 더 작은 learning rate을 사용한다. 하지만 비선형적 입력의 분포가 학습 시 더 안정적으로 유지된다면 optimizer는 기울기 포화에 빠질 가능성이 줄어들고 학습은 가속될 것이다.

연구진은 학습 과정에서 심층신경망의 내부 노드의 분포가 변하는 것을 ***Internal Covariate Shift***라고 정의하였다. 이것을 제거하는 것은 학습이 빨라지도록 한다. 연구진은 internal covariate shift를 제거하고 학습 속도를 급격히 빠르게 만드는 ***Batch Normalization***이라는 새로운 메커니즘을 제안한다. 이것은 normalization 단계를 통해 layer 입력의 평균과 표준편차를 고정함으로써 이루어진다. Batch Normalization은 network 전체의 gradient 흐름에도 이점이 있는데, parameter의 scale이나 초깃값에 대한 gradient의 의존도를 감소시킨다. 이로 인해 발산의 위험성 없이 더 큰 learning rate을 사용할 수 있게 된다. 게다가 Batch Normalization은 모델을 규제하여 Dropout의 필요성을 감소시킨다. 마지막으로 기울기의 포화도 예방한다.

§4.2에서 현재 가장 뛰어난 ImageNet 분류 network에 Batch Normalization을 적용하였고 그 결과 단 7%의 학습 단계만에 동등한 성능을 낼 수 있었다. Batch Normalization을 사용한 network를 앙상블하여 알려진 최고 수준의 top-5 error rate을 달성하였다.
</br>
</br>

## 2. Towards Reducing Internal Covariate Shift
입력이 whitened(평균 0, 분산이 1을 갖고 상관성이 없도록 선형 변환하는 것을 whitening 이라고 표현) 되었을 때 network의 학습이 빨라진다는 것은 오래전부터 알려져 있었다. 각 layer의 입력을 whitening함으로써 분포를 고정하고 internal covariate shift의 나쁜 효과를 제거하는데에 한 발 다가가고자 한다.

연구진은 매 학습 단계 또는 특정한 간격으로 network를 직접 수정하거나 parameter를 변경함으로써 whitening activation을 고려하였다. 그러나 이러한 수정들이 optimization 단계에 배치된다면, gradient descent 단계에서 parameter는 normalization의 업데이트를 요구하는 방향으로 개선될 것이고, 이는 gradient 단계의 효과를 감소시킨다.

위의 접근법의 문제는 gradient descent optimization이 normalization이 일어난다는 점을 고려하지 않는다는 것이다. 이 문제를 해결하기 위해, 임의의 parameter 값에 대해 network가 항상 이상적인 활성화를 출력하도록 보장하고자 한다. 그렇게 함으로써 loss의 gradient가 normalization과 모델 parameter의 의존성을 설명할 수 있게 된다. 

x를 layer 입력 벡터, X를 학습데이터셋의 x의 집합이라고 할 때 normalization은 Norm(x, X)로 쓸 수 있다. 즉 입력 x 뿐만 아니라 parameter에 의존하는 모든 X에 의존하게 된다. 이와 같은 framework에서 layer 입력을 whitening 하는 것은 많은 계산량을 요구한다. 따라서 연구진은 미분이 가능하며, 매 parameter 업데이트마다 전체 학습 데이터셋의 분석을 필요로 하지 않는 새로운 normalization 방법을 탐색하였다. 

기존의 몇몇 접근법들은 하나의 학습 샘플만을 이용하거나, 이미지의 경우 동일한 위치의 여러 샘플을 이용하였다. 하지만 이러한 방법은 activation의 절대적 스케일을 버림으로써 표현력을 바꾸게 된다. 연구진은 전체 학습 데이터의 통계와 관련하여 학습 샘플을 normalization함으로써 신경망의 정보들을 보존하고자 한다.
</br>
</br>

## 3. Normalization via Mini-Batch Statistics
먼저, layer 입력과 출력의 feature들을 함께 whitening 하는 대신, 각 scalar feature를 독립적으로 normalize 한다. d차원 입력 x에 대해 각 차원을 다음과 같이 normalize하며 기댓값과 분산은 전체 학습 데이터셋을 통해 계산된다.
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/160398672-a4ce55d3-9c08-40ed-af99-f01cf2a2b011.png" width="20%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/160398701-36a091a4-d719-4745-b60e-d3f7941d0403.png" width="20%"></p>

단순히 각 layer의 입력을 normalizing하는 것은 해당 layer의 표현력을 변화시킬 수 있다. 이 문제를 해결하기 위해, 신경망에 추가된 변환이 identity transform을 대신할 수 있도록 한다. 그렇게 하기 위해 연구진은 normalize된 값을 scale 및 shift 하는 parameter, gamma와 beta를 도입한다.
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/160400594-03256a41-e3dc-4273-a922-81ffbb024005.png" width="20%"></p>

이 parameter들은 원래의 모델 parameter와 함께 학습되며 신경망의 표현력을 저장한다. 사실 gamma^(k) = sqrt(Var[x^(k)]), beta^(k) = E[x^(k)]로 설정하면 기존 activation을 복구할 수 있다.

우리는 mini-batch SGD를 사용하므로, mini-batch가 각 activation의 평균 및 분산의 추정량을 만들어낸다고 하자. 그러면 normalization에 사용한 통계량이 gradient backpropagation에 완전히 참여할 수 있다. mini-batch의 사용은 결합공분산이 아닌 차원별 분산의 계산 덕에 가능함을 주목한다.

크기 m의 mini-batch B를 생각하자. 어떤 activation x^(k)에 대해 k를 떼어두면 mini-batch에는 m개의 activation이 존재한다. 정규화된 값을 x_hat, 변환된 값을 y라고 하면 Batch Normalizing Transform은 다음과 같다. 
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/160408202-dd1b61ca-6a95-4739-8591-283a522b6299.png" width="50%"></p>

Batch Normalization은 학습 샘플 x와 mini-batch 내의 다른 학습 샘플들에 의존한다. normalized activation x^k_hat은 변환의 내부에 있지만 그 존재는 매우 중요하다. 각 x^k_hat은 y = gamma * x + beta로 구성된 sub-network의 입력으로 볼 수 있으며 그 다음에는 원래의 network의 다른 처리가 뒤따른다. 따라서 그 입력은 모두 고정된 평균과 분산을 갖게 되며, 비록 결합분포는 학습 과정 중 변할 수 있지만 우리는 이로 인해 학습이 가속화 될 것이라고 예상한다. 






