> # Rethinking the Inception Architecture for Computer Vision
> ### Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna (2015)
</br>

## 1. Introduction
AlexNet은 2012 ImageNet 대회에서 우승한 이후, object-detection, segmentation, human pose estimation, video classification, object tracking, superresolution과 같은 다양한 컴퓨터비전 분야에 성공적으로 적용되어 왔다. 그 이후 VGGNet이나 GoogLeNet과 같은 network들도 등장하여 높은 성능을 보여주었다. 

비록 VGGNet은 구조적 단순함이라는 강력한 특징을 갖고 있지만 계산 비용이 너무 높다. 반면 GoogLeNet의 Inception architecture는 메모리와 계산 비용을 제한하고도 좋은 성능을 보인다. GoogLeNet은 AlexNet보다 12배 적은 parameter를 사용하는데 VGGNet은 AlexNet보다 3배나 많은 parameter를 사용한다. 

그럼에도 불구하고, Inception architecture의 복잡도 때문에 network를 변화시키기 어렵다. architecture의 규모를 단순하게 키운다면, 계산적 이득의 많은 부분이 즉시 소실된다. 또한, 앞선 논문은 GoogLeNet의 디자인에 기여한 요소들에 대한 분명한 설명을 제시하지 않는다. 때문에 network의 효율성을 유지하면서 새로운 케이스에 적응시키는 것이 쉽지 않다. 본 논문에서는 효율적으로 convolution network의 규모를 키우는 데에 유용하다고 알려진 몇 가지 일반적 원리와 최적화 아이디어를 설명한다. 
</br>
</br>

## 2. General Design Principles
아래 원리들의 유용함은 추측에 근거한 것이며, 유효성과 정확도를 평가하기 위해 추가적인 실험적 증거가 필요할 것이다. 

1. representational bottleneck을 피해라, 특히 network의 초기에. feed-forward network는 input layer로부터 마지막 classifier 또는 regressor로 이어지는 비순환적 그래프로 표현할 수 있다. input에서 output으로의 흐름을 나누는 어떠한 cut이든, 그 cut을 통해 정보의 양에 접근할 수 있다????? 우리는 강하게 압축하는 bottleneck을 피해야 한다. 일반적으로 representation의 사이즈는 output에 가까워짐에 따라 부드럽게 감소해야 한다. 이론적으로, 정보는 단순히 representation의 차원에 의해 평가될 수는 없는데, 이는 correlation structure와 같은 중요한 요소들이 버려졌고, 차원은 단지 정보의 rough한 추정치이기 때문이다.

2. 더 높은 차원의 representation이 network 내에서 지역적으로 처리하기 더 쉽다. convolutional network의 tile마다 activation을 증가시키는 것은 더 많은 disentangled feature를 가능하게 한다.

3. 낮은 차원의 embedding을 통해 표현력의 손실 없이(또는 거의 없이) spatial aggregation이 가능하다. 우리는 그 이유가, 인접한 유닛들 사이의 강한 상관관계가 차원 감소 동안 훨씬 적은 정보의 손실로 이어졌기 때문이라고 가설을 세웠다. 이러한 신호들은 쉽게 압축할 수 있기 때문에 차원의 감소는 학습 속도 또한 높여준다. 

4. network의 너비와 깊이의 균형을 맞춘다. 각 단계의 필터 수와 network의 깊이 사이의 균형을 맞춤으로써 최적의 성능에 도달할 수 있다. network의 깊이와 너비를 늘리는 것 모두 성능을 높이는데에 기여한다. 하지만 최적의 성능 개선은 그 둘이 병렬적으로 증가할 때에 이룰 수 있다. 따라서 computational budget은 network의 깊이와 너비에 고르게 분배되어야 한다.

비록 이런 원리들이 성립하더라도, network의 성능 개선을 위해 이를 사용하는 것은 간단하지 않다. 아이디어는 애매한 상황에서만 신중하게 사용하자는 것이다.

## 3. Factorizing Convolutions with Large Filter Size
GoogLeNet network의 이점의 많은 부분은 dimension reduction의 넉넉한 사용으로부터 발생한다. 이것은 효율적인 계산을 위해 convolution을 분해하는 것의 특별한 케이스로 볼 수 있다.

여기서 우리는 계산의 효율성을 높이기 위한 다른 convolution 분해 방법을 탐색한다. Inception network는 fully convolutional이기 때문에, 각 가중치는 activation 당 하나의 연산에 대응된다. 그러므로 계산 비용의 감소는 parameter 수의 감소로 이어진다. 이것은, 적절한 분해를 통해 우리는 더 disentangled parameter를 얻을 수 있고 결국 더 빠른 학습을 이뤄낼 수 있음을 의미한다.

### 3.1 Factorization into smaller convolutions
큰 필터를 사용하는 convolution(e.g. 5x5, 7x7)은 계산비용이 불균형하게 높은 경향이 있다. 예를 들어, 5x5 convolution은 3x3에 비해 계산량이 2.78배 높다. 물론 5x5 필터는 초기 layer에서 더 먼 거리의 신호 간의 의존성을 포착할 수 있다. 하지만 우리는 5x5 convolution을, 더 적은 parameter를 갖는 multi-layer (mini) network로 대체할 수 있는지 묻고 싶다. 5x5 convolution의 계산을 들여다보면 이는 두 개의 layer를 가진 3x3 convolution으로 대체할 수 있다. 
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/163204531-c7cc39ab-a0db-417b-b854-90daab45c17c.png" width="40%"></p>

이러한 세팅은 인접한 타일 간에 weight를 공유하면서 parameter의 수를 줄여준다. 계산비용 절감에 대해 분석하기 위해, 일반적인 상황에 대한 몇 가지 단순화 가정을 설정한다. 우리는 n = alpha·m 이라고 가정하고, 상수 alpha에 의해 activation의 수를 변경하고자 한다. 5x5 convolution이 집계를 하므로, alpha는 일반적으로 1보다 조금 크다(GoogLeNet의 경우 1.5). 5x5 layer를 2개의 layer로 대체할 때, 이 확장은 두 단계로 나누어 진행된다: 각 단계에서 필터의 수를 sqrt(alpha)만큼 늘린다. 두 개의 3x3 convonlutional layer로 대체하면, 계산량을 28% 줄일 수 있다. 

### 3.2 Spatial Factorization into Asymmetric Convolutions
위의 결과에 따르면 3x3보다 큰 필터의 convolution은 여러 개의 3x3 convolutional layer로 대신할 수 있으므로 일반적으로 덜 효율적이다. 그러면 더 작은, 예를 들어 2x2 convolution으로 분해하는 것이 더 좋은가에 대한 의문을 제기할 수 있다. 하지만 2x2 보다는 nx1과 같은 비대칭 convolution이 더 효과적인 것으로 밝혀졌다. 예를 들어, 3x1과 1x3 convolution을 차례로 사용하는 것은 3x3 convolution의 two-layer network와 동등하다. 하지만 input과 output 필터의 수가 같다면 two-layer의 계산비용이 33% 더 저렴하다. 3x3 convolution을 2개의 2x2 convolution으로 분해하면 계산량을 11%만 감소시킬 수 있다.

이론적으로는, 모든 nxn convolution을 1xn -> nx1 convolution으로 할 수 있고, n이 커짐에 따라 계산량은 급격히 감소하게 된다. 그러나 실제로는, 초기 layer에서는 이러한 분해가 잘 동작하지 않는다는 것을 알아냈다. 하지만 적당한 사이즈(mxm feature map에 대해 m이 12~20일 때)에서는 좋은 결과를 낸다. 이때는 1x7과 7x1 convolution을 사용하여 매우 좋은 성능을 얻을 수 있다. 

## 4. Utility of Auxiliary Classifiers
Inception architecture에서 auxiliary classifier를 제안했다. 그 목적은 더 낮은 layer에서 유용한 gradient를 푸시하고 vanishing gradient에 맞서 수렴을 돕는 것이다. 흥미롭게도 연구진은 auxiliary classifier가 학습의 빠른 수렴에는 도움이 되지 않는다는 것을 발견했다. network가 최고 정확도에 도달할 때까지 auxiliary classifier가 있든 없든 그 학습은 거의 동일하다. 학습이 막바지에 다다르면 auxiliary classifier를 적용한 network의 성능이 앞서나가고 최고점에 도달한다. 












