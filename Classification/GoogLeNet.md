> # Going Deeper with Convolutions (2014)
> ## Christian Szegedy et al.

</br>

## 1. Introduction
본 논문에서는 모델을 디자인할 때, 순전히 accuracy numbers (error rate나 정확도 얘기하는 듯?)에 집착하기보다는 메모리 사용량과 같은 알고리즘 효율성을 고려하였다.
대부분의 실험에서 연산량의 한계를 정해두었기 때문에 단순히 학문적인 관점 뿐만 아니라 현실에서도 합리적인 비용으로 사용할 수 있다.

본 논문에서 'deep' 이라는 용어는 두 가지 의미로 사용된다:
- 일반적인 의미 (신경망의 층이 깊어지는 것)
- Inception 모듈 형태의 조직의 새로운 레벨 ????
</br>
</br>

## 2. Related Work
LeNet-5 이후, CNN 모델은 몇 층의 convolutional layer와 normalization, max-pooling과 fully-connected layer로 이루어진 정형화된 틀을 갖게 되었다. 요즘 트렌드는 layer의 수를 더 늘리고 overfitting을 방지하기 위해 dropout을 사용하는 것이다.

Network-in-Network는 Lin[12]에 의해 제안된 방식으로 1x1 convolutional layer를 추가힌다. 본 연구에서도 해당 방식을 많이 사용한다. 하지만 여기에서 1x1 convolution은 두 가지 목적이 있다. 우선 이것은 computational bottleneck을 제거하기 위한 차원 감소 역할을 한다. 또한 심각한 성능 저하 없이 모델의 깊이와 너비를 증가시킨다. 

마지막으로 현재 object detection 분야의 SOTA인 R-CNN은 모든 detection 문제를 두 가지 하위 단계로 분해한다 - location을 위해 색깔과 질감 같은 low level 신호를 처리하고 해당 위치의 물체 식별을 위해 CNN classifier를 사용한다. 본 논문에서도 유사한 방식을 사용하되 이를 좀 더 발전시켰다. bounding box 예측에 multi-box predictino을 사용하고 분류를 위해 앙상블을 적용하였다.
</br>
</br>

## 3. Motivation and High Level Considerations
모델의 성능을 높이는 가장 간단한 방법은 모델의 depth와 width를 늘리는 것이다. 그러나 이는 overfitting과 연산량 증가라는 문제를 야기한다. 
해당 문제들을 해결하기 위한 근본적인 방법은 sparsity를 도입하고 심지어 fully-connected layer도 sparse한 것으로 대체하는 것이다. 
Arora[2]에 의하면 데이터셋의 확률분포가 아주 크고 희박한 심층신경망으로 표현된다면, 최적의 network topology는 이전 층의 활성화 함수들의 상관관계를 분석하고 상관관계가 높은 뉴런들을 군집화하여 층별로 구성할 수 있다.

불행하게도 현재의 컴퓨팅 시스템은 non-uniform sparse 데이터의 연산에 매우 비효율적이다.
Inception architecture는 sparse한 구조를 근사화하는 정교한 network topology 구축 알고리즘의 가상 출력을 평가하고, dense하고 쉽게 사용될 수 있는 구성요소로 가설 세워진 출력을 커버하는 사례 연구로 시작되었다. Inception architecture의 성공에도 불구하고 그 구축을 이끈 지도 원리에 기여할 수 있는지 여전히 의문이다.
</br>
</br>

## 4. Architectural Details
본 논문에서는 우리가 해야할 일은 최적의 local construction을 찾고 이를 공간적으로 반복하는 것이라고 말한다. 
Arora[2]는 마지막 layer의 상관관계를 분석하고 높은 상관성을 띄는 유닛들끼리 묶는 layer-by-layer 구조를 제안한다. 이런 cluster들은 이전 layer와 연결되어 다음 layer의 유닛을 형성한다. 이전 층의 유닛들이 input 이미지의 특정 지역에 대응되고 해당 유닛들은 filter bank로 나누어질 것이라고 추정한다. 더 낮은 층의 유닛들은 local region에 집중할 것이다. 따라서 결국 많은 cluster들이 하나의 region에에 집중할 것이고 그것들은 다음 층의 1x1 convolutional layer에 의해 커버될 것이다.


