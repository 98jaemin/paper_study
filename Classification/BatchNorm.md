> # Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (2015)
> ### Sergey Ioffe, Christian Szegedy
</br>

## 1. Introduction
딥러닝은 vision과 speech 등 다양한 분야에서 급속도로 발전하고 있다. 확률적 경사하강법(SGD)는 심층신경망을 학습시키는 효율적인 방식임이 증명되었고, momentum이나 Adagrad와 같은 변형들도 사용되고 있다. SGD의 각 학습 단계에서 크기가 m인 mini-batch를 고려한다. mini-batch는 parameter에 대한 loss function의 gradient를 근사하기 위해 사용된다.

<p align="center"><img src="https://user-images.githubusercontent.com/86872735/159493642-794bb5cc-39e9-42b8-9234-7c076a42425c.png" width="20%"></p>

한번에 하나의 데이터를 사용하는 것과 비교하여 mini-batch를 사용하는 것은 몇 가지 이점이 있다.
- mini-batch loss의 gradient는 전체 학습 데이터의 gradient의 추정값이며 이는 batch size가 커질수록 개선된다
- 각 데이터에 대해 m번 연산하는 것보다 batch를 통해 연산하는 것이 훨씬 효율적이다. (parallelism)

각 layer의 입력은 모든 이전 layer의 parameter에 영향을 받게 되며, 따라서 신경망이 깊어질수록 작은 변화도 증폭되게 된다.

layer의 입력의 분포의 변화는 layer가 계속해서 새로운 분포에 적응해야 한다는 문제를 야기한다. 신경망이 다음과 같은 연산을 한다고 하자. F_1과 F_2는 임의의 변환이고 Theta_1과 Theta_2는 loss를 최소화하기 위해 학습되는 parameter이다.
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/159498055-8b797b0f-7fb9-4983-be10-ecddcb602b11.png" width="30%"></p>

Theta_2에 대한 학습은 x=F_1(u, Theta_1)일 때 다음과 같다.
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/159498647-3822f745-41c3-438f-a8be-896bc46b4f76.png" width="20%"></p>

gradient descent 단계에서 연산을 할 때, 이는 신경망 F_2에 입력 x를 받아들이는 것과 동등하다. 입력 분포의 특성은 sub-network에도 적용된다. x의 분포가 고정된다면 Theta_2는 매번 변화에 맞추어 재조정될 필요가 없다.

입력의 고정된 분포는 sub-network 바깥에도 그 이점이 있다. sigmoid 활성화 함수를 생각해보자. 0 근처가 아닌 모든 x에 대해 기울기 소실 문제가 발생하고 학습이 느리게 이루어진다. 이러한 포화 문제를 해결하기 위해 일반적으로 초기화를 조정하거나 ReLU, 또는 더 작은 learning rate을 사용한다. 하지만 비선형적 입력의 분포가 학습 시 더 안정적으로 유지된다면 optimizer는 기울기 포화에 빠질 가능성이 줄어들고 학습은 가속될 것이다.

연구진은 학습 과정에서 심층신경망의 내부 노드의 분포가 변하는 것을 ***Internal Covariate Shift***라고 명명하였다. 이것을 제거하는 것은 학습이 빨라지도록 한다. 연구진은 internal covariate shift를 제거하고 학습 속도를 급격히 빠르게 만드는 ***Batch Normalization***이라는 새로운 메커니즘을 제안한다. 이것은 normalization 단계를 통해 layer 입력의 평균과 표준편차를 고정함으로써 이루어진다. Batch Normalization은 또한 network 전체의 gradient 흐름에도 이점이 있는데, parameter의 scale이나 초깃값에 대한 gradient의 의존도를 감소시킨다. 이로 인해 발산의 위험성 없이 더 큰 learning rate을 사용할 수 있게 된다. 


