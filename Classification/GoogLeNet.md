> # Going Deeper with Convolutions (2014)
> ## Christian Szegedy et al.

</br>

## 1. Introduction
GoogLeNet은 2년 전 우승했던 AlexNet보다 12배 적은 parameter를 사용하면서도 훨씬 좋은 성능을 냈다. 
본 논문에서는 모델을 디자인할 때, 정확도 수치에 집착하기보다는 메모리 사용량과 같은 알고리즘 효율성을 고려하였다.
대부분의 실험에서 연산량의 한계를 정해두었기 때문에 단순히 학문적인 관점 뿐만 아니라 현실에서도 합리적인 비용으로 사용할 수 있다.

본 논문에서 'deep' 이라는 용어는 두 가지 의미로 사용된다:
- 일반적인 의미 (신경망의 층이 깊어지는 것)
- Inception 모듈이라는 새로운 구조의 도입
</br>

## 2. Related Work
LeNet-5 이후, CNN 모델은 몇 층의 convolutional layer와 normalization, max-pooling과 fully-connected layer로 이루어진 정형화된 틀을 갖게 되었다. 요즘 트렌드는 layer의 수를 더 늘리고 overfitting을 방지하기 위해 dropout을 사용하는 것이다.

Network-in-Network는 Lin[12]에 의해 제안된 방식으로 1x1 convolutional layer를 추가힌다. 본 연구에서도 해당 방식을 많이 사용한다. 하지만 여기에서 1x1 convolution은 두 가지 목적이 있다. 우선 이것은 computational bottleneck을 제거하기 위한 차원 감소 역할을 한다. 또한 큰 성능 저하 없이 모델의 깊이와 너비를 증가시킨다. 

마지막으로 현재 object detection 분야의 SOTA인 R-CNN은 모든 detection 문제를 두 가지 하위 단계로 분해한다
- 색깔이나 질감과 같은 low level feature를 이용하여 category에 구애받지 않는 방식으로 object location proposal을 생성
- CNN classifier를 이용해 해당 위치의 물체 category 분류
 
이러한 two stage approach는 low-level feature를 활용한 bounding box segmentation의 정확성 뿐만 아니라, state-of-the-art CNN의 강력한 classification power를 활용할 수 있다.
본 논문의 detection submission에도 이와 유사한 pipeline을 적용했지만, 두 단계 모두에 대한 개선 사항이 있었다.
- object bounding box의 높은 recall을 위한 multi-box prediction
- 분류를 위해 앙상블
</br>

## 3. Motivation and High Level Considerations
모델의 성능을 높이는 가장 간단한 방법은 모델의 depth와 width를 늘리는 것이다. 그러나 이는 overfitting과 연산량 증가라는 문제를 야기한다. 
해당 문제들을 해결하기 위한 근본적인 방법은 fully-connected layer와 convolution 내부를 sparse한 것으로 대체하여 sparsity를 도입하는 것이다. 
Arora[2]에 의하면 데이터셋의 확률분포가 아주 크고 희박한 심층신경망으로 표현된다면, 최적의 network topology는 이전 층의 활성화 함수들의 상관관계를 분석하고 상관관계가 높은 뉴런들을 군집화하여 층별로 구성할 수 있다.

불행하게도 현재의 컴퓨팅 시스템은 non-uniform sparse 데이터의 연산에 매우 비효율적이다.
sparsity를 도입하여 산술 연산의 수가 100배 감소하더라도, 참조하면서 발생되는 lookup이나 cache miss로 인한 오버 헤드가 이를 상회하는 역효과를 낳는다.
또한, CPU나 GPU에 최적화 된 numerical library가 개발되면서 dense matrix multiplication의 고속 연산이 가능해졌고, 이에 따라 operation의 수가 줄어듦으로 인한 이득이 점점 감소했다

Inception architecture는 sparse structure에 대한 근사화를 포함해, dense하면서도 쉽게 사용할 수 있도록 정교하게 설계된 network topology construction 알고리즘을 평가하기 위한 사례 연구로 시작되었다. Inception architecture의 성공에도 불구하고 그 구축을 이끈 지도 원리에 기여할 수 있는지 여전히 의문이다.
</br>
</br>

## 4. Architectural Details
본 논문에서는 우리가 해야할 일은 최적의 local construction을 찾고 이를 공간적으로 반복하는 것이라고 말한다. 
Arora[2]는 마지막 layer의 상관관계를 분석하고 높은 상관성을 띄는 유닛들끼리 묶는 layer-by-layer 구조를 제안한다. 이런 cluster들은 이전 layer와 연결되어 다음 layer의 유닛을 형성한다. 이전 층의 유닛들이 input 이미지의 특정 지역에 대응되고 해당 유닛들은 filter bank로 나누어질 것이라고 가정한다. 더 낮은 층의 유닛들은 local region에 집중할 것이다. 따라서 결국 많은 cluster들이 하나의 region에에 집중할 것이고 그것들은 다음 층의 1x1 convolutional layer에 의해 커버될 것이다. 이때 patch-alignment issue\*를 피하기 위해 Inception architecture에서는 1x1, 3x3, 5x5 사이즈의 필터를 사용한다. 제안된 architecture는 다음 단계의 input을 형성하는 하나의 output으로 연결된 output filter banks와 layer들의 조합이다.

이러한 "Inception module"이 겹겹이 쌓여있기 때문에, 그 output은 달라지기 마련이다. high layer에서 high-level feature를 추출한다면 spatial concentration은 감소할 것이다. 이는 3x3, 5x5 convolution이 더 필요함을 의미한다. 그러나 5x5 convolution을 몇 개만 사용해도 계산비용이 매우 높아지는 문제가 있다. 따라서 이러한 구조가 최적의 sparse structure를 커버한다 하더라도 매우 비효율적인 연산을 하게 된다.

따라서 계산량 감소를 위해 차원을 줄이는 방법을 사용한다. 표현은 대부분의 위치에서 sparse해야 하고 전체적으로 집계되어야 할 때에만 신호를 압축해야 한다.  즉, 3x3 convolution과 5x5 convolution 이전에 1x1 convolution이 사용된다.

Inception network는 앞서 이야기한 모듈과 stride가 2인 max-pooling layer로 구성된 network이다. 학습 시 메모리 문제 때문에 낮은 layer에서는 전통적인 convolution을 사용하고 높은 layer에서만 Inception module을 사용하는 것이 좋은 듯하다.

이 architecture의 유용한 점은 이후 단계에서의 걷잡을 수 없는 계산량 증가 없이 unit의 수를 많이 늘릴 수 있다는 것이다. 이는 큰 사이즈의 convolution 앞에 차원 감소를 사용한 덕분이다. 뿐만 아니라 그 디자인은 시각 정보는 다양한 스케일로 처리되고 종합됨으로써 다양한 스케일에서 동시에 feature를 추출한다는 실용적인 직관을 따른다.

계산자원의 향상된 사용 덕분에 계산적 어려움 없이 각 단계의 너비와 단계의 수를 늘릴 수 있게 된다. 연구진들은 정교한 조절을 통해 Inception architecture를 사용하지 않는 비슷한 성능의 다른 모델보다 3~10배 빠른 모델을 구현할 수 있음을 확인했다. 

\* patch-alignment issue : filter size가 짝수일 경우 patch의 중심을 어디로 두어야 할지도 정해야 하는 문제
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

지난 몇 달간, 이미지 샘플링 방법은 크게 달라졌다. 따라서 본 network를 학습시키는 확정적인 가이드를 제공하기는 어렵다. 설상가상으로, 몇몇 모델은 상대적으로 작은 crop을, 몇몇은 큰 crop을 이용해 학습했다. 그러나 아주 잘 동작하는 것으로 확인된 한 방법은 8%에서 100%까지 분포된 다양한 사이즈의 patch를 샘플링하는 것이다. 또한 연구진은 광학적 왜곡이 overfitting 예방에 유용하다는 것을 알아냈다.

<p align='center'><img src='https://user-images.githubusercontent.com/86872735/156366401-86a90567-f058-479e-aa58-3e7587d05713.png'></p>
</br>

## 7. ILSVRC 2014 Classification Challenge Setup and Results
연구진은 총 7가지 버전의 GoogLeNet 모델을 학습시킨 뒤 이를 앙상블하여 예측에 활용하였다. 모델들은 동일한 초기화 방법과 learning rate을 사용하였으며 샘플링 방법과 input 이미지 순서만 달리했다. 

test 단계에서 적극적인 cropping이 사용되었다. 먼저 이미지의 짧은 변이 256, 288, 320, 352 중 하나가 되도록 resize를 한 후 왼쪽, 가운데, 오른쪽 중 하나를 고른 뒤, 다시 네 코너와 가운데 중 하나를 224x224 크기로 resize하였다(+ 좌우 반전). 충분한 수의 crop이 존재한다면 더 많은 crop의 이득은 미미하기 때문에 이러한 적극적인 cropping은 현실의 문제에서는 반드시 필요하지는 않다.

최종 예측을 얻기 위해 각 classifier와 crop의 softmax 함수값을 평균내었다. 
</br>
</br>

## 9. Conclusions
본 연구는 쉽게 사용 가능한 dense building block으로 sparse structure를 근사화하는 것이 컴퓨터비전의 신경망을 개선하는 실행 가능한 방법이라는 것의 확고한 증거를 제시한다. 해당 방법의 가장 큰 장점은 더 얕고 좁은 architecture에 비해 약간의 계산량을 더함에도 상당한 성능 개선을 얻는다는 점이다. 

본 연구의 object detection은 context를 사용하거나 bounding box regression을 수행하지 않았음에도 불구하고 경쟁력이 있었으며, 이는 Inception architecure의 성능에 대한 증거를 보여주었다.

classification과 detection 모두에서, 비슷한 깊이와 너비의 훨씬 더 계산비용이 높은 non-Inception network도 비슷한 결과를 얻을 것으로 예상된다. 그러나 본 연구는 일반적으로 더 sparse한 architecture를 이용하는 것이 실현 가능하고 유용한 아이디어라는 명백한 증거를 제시한다. 
