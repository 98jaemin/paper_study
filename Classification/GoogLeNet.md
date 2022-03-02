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
Arora[2]는 마지막 layer의 상관관계를 분석하고 높은 상관성을 띄는 유닛들끼리 묶는 layer-by-layer 구조를 제안한다. 이런 cluster들은 이전 layer와 연결되어 다음 layer의 유닛을 형성한다. 이전 층의 유닛들이 input 이미지의 특정 지역에 대응되고 해당 유닛들은 filter bank로 나누어질 것이라고 추정한다. 더 낮은 층의 유닛들은 local region에 집중할 것이다. 따라서 결국 많은 cluster들이 하나의 region에에 집중할 것이고 그것들은 다음 층의 1x1 convolutional layer에 의해 커버될 것이다. 이때 patch-alignment issue를 피하기 위해 Inception architecture에서는 1x1, 3x3, 5x5 사이즈의 필터를 사용한다. 제안된 architecture는 다음 단계의 input을 형성하는 하나의 output으로 연결된 output filter banks와 layer들의 조합이다.

이러한 "Inception module"이 겹겹이 쌓여있기 때문에, 그 output은 달라지기 마련이다. high layer에서 high-level feature를 추출한다면 spatial concentration은 감소할 것이다. 이는 3x3, 5x5 convolution이 더 필요함을 의미한다. 그러나 5x5 convolution을 몇 개만 사용해도 계산비용이 매우 높아지는 문제가 있다. 따라서 이러한 구조가 최적의 sparse structure를 커버한다 하더라도 매우 비효율적인 연산을 하게 된다.

따라서 계산량 감소를 위해 차원을 줄이는 방법을 사용한다. 표현은 대부분의 위치에서 sparse해야 하고 전체적으로 집계되어야 할 때에만 신호를 압축해야 한다.  즉, 3x3 convolution과 5x5 convolution 이전에 1x1 convolution이 사용된다.

Inception network는 앞서 이야기한 모듈과 stride가 2인 max-pooling layer로 구성된 network이다. 학습 시 메모리 문제 때문에 낮은 layer에서는 전통적인 convolution을 사용하고 높은 layer에서만 Inception module을 사용하는 것이 좋은 듯하다.

이 architecture의 유용한 점은 이후 단계에서의 걷잡을 수 없는 계산량 증가 없이 unit의 수를 많이 늘릴 수 있다는 것이다. 이는 큰 사이즈의 convolution 앞에 차원 감소를 사용한 덕분이다. 뿐만 아니라 그 디자인은 시각 정보는 다양한 스케일로 처리되고 종합됨으로써 다양한 스케일에서 동시에 feature를 추출한다는 실용적인 직관을 따른다.

계산자원의 향상된 사용 덕분에 계산적 어려움 없이 각 단계의 너비와 단계의 수를 늘릴 수 있게 된다. 연구진들은 정교한 조절을 통해 Inception architecture를 사용하지 않는 비슷한 성능의 다른 모델보다 3~10배 빠른 모델을 구현할 수 있음을 확인했다. 
</br>
</br>

## 5. GoogLeNet
모델의 모든 convolution은 ReLU를 사용한다. 평균이 0인 224x224 크기의 RGB 이미지를 입력 받는다. '#3x3 reduce'와 '#5x5 reduce'는 3x3과 5x5 convolution 이전에 사용된 1x1 차원 감소 filter의 개수를 의미한다. pool proj column은 built-in max-pooling 이후 projection layer에서의 1x1 filter 개수이다. 모든 reduction/projection layer에서도 ReLU를 사용한다.

<p align="center"><img src="https://user-images.githubusercontent.com/86872735/156185614-f637f397-4a80-4657-9f97-a6d2cf5dbf04.png"></p>

모델은 22개의 layer로 이루어져 있다(pooling을 포함하면 27개). fully-connected layer를 average pooling으로 바꾸면 0.6%의 top-1 성능 개선이 이루어지는 것을 확인했지만 dropout은 여전히 반드시 필요했다.

꽤 깊은 모델이기 때문에, 기울기 역전파의 효율이 걱정이었다. 동일한 task를 해결하던 얕은 모델들을 보면 모델의 중간부 layer들이 매우 차별적이어야한다. 중간층에 auxiliary classifier를 추가하여 낮은 층에서의 차별화를 도모한다. 이것은 regularization을 제공하며 gradient vanishing을 방지할 것으로 생각된다. auxiliary classifier들은 Inception (4a)와 (4d) 모듈의 출력 위에 놓인 convolutional network가 축소된 형태를 취한다. 학습동안 그것들의 loss는 discount weight 없이 total loss에 더해진다. inference에서는 사용되지 않는다. 

auxiliary classifier의 network 구조는 다음과 같다:
- average pooling : 5x5 filter, stride 3
  - 4x4x512 output -> Inception (4a)
  - 4x4x528 output -> Inception (4d)
- convolution : 1x1 filter 128개
  - ReLU
- fully connected : 1024 units
  - ReLU
- dropout : 0.7
- softmax : training에서만 사용

## 6. Training Methodolohy
학습에는 momentum 0.9인 비동기적 SGD를 사용하였고 learning rate은 8 epoch마다 4%씩 감소시켰다. 또한 추론 시의 마지막 모델 생성을 위해 Polyak averaging을 사용했다. 
