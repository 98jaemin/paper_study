> # Very Deep Convolutional Networks For Large-Scale Image Recognition (2014)
> ### Karen Simonyan, Andrew Zisserman
</br>

## 1. Introduction
CNN이 컴퓨터비전 분야에서 점점 더 사용되면서, 기존의 구조를 개선하여 성능을 높이려는 많은 노력들이 이루어졌다. 본 논문에서는 그 중 모델의 depth 문제를 다룬다. architecture의 다른 parameter는 고정하고, convolutional layer를 추가하여 network의 깊이만 조금씩 늘려간다. 이는 모든 layer에서 3x3 크기의 매우 작은 filter를 사용하기 때문에 가능하다.

결과적으로, ILSVRC에서 가장 좋은 성능을 냈을 뿐만 아니라 다른 dataset에도 적용 가능한 매우 정확한 CNN architecture를 얻을 수 있었다. 
</br>

## 2. ConvNet Configurations
### 2.1 Architecture
학습에는 224x224 사이즈의 RGB 이미지를 사용한다. 각 픽셀의 RGB 평균값을 빼는 것이 유일한 전처리 작업이다. convolutional layer에서 3x3의 작은 사이즈의 필터를 사용한다. input channel의 선형 변환을 위한 1x1 필터도 사용한다. convolution의 stride는 1로 고정하고 padding은 1로 설정한다. 몇몇 convolutional layer 뒤에는 filter size 2, stride 2인 max pooling layer를 추가한다.

convolution block에 이어 3개의 fully connected layer가 존재한다. 첫 번째, 두 번째는 각각 4096 채널이고 세 번째는 1000개의 채널을 갖는다. 마지막 layer는 softmax이다. 

모든 hidden layer는 ReLU를 사용한다. 하나의 네트워크를 제외하고는 Local Response Normalization을 사용하지 않는다.

### 2.2 Configurations
각 network의 구성은 Table 1과 같다. 모두 구성은 동일하되 depth만 다르다. convolutional layer의 너비는 64에서 시작하여 max pooling layer 이후에 2의 거듭제곱으로 증가하여 512까지 늘어난다.
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/158050994-883e375b-7fb8-4d46-9b35-b1a8895bae69.png"></p>
</br>

Table 2에는 각각의 parameter 수를 나타내었다. 높은 depth에도 불구하고 더 얕고 wide한 모델에 비해 parameter 수가 높지 않다.
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/158051822-74124313-ac0b-463f-89e4-f6fe8e3fe007.png"></p>
</br>

### 2.3 Discussion
본 연구의 network는 기존의 상위권 network와 사뭇 다르다. 첫 번째 layer에서 상대적으로 큰(e.g. 11x11 with stride 4, 7x7 with stride 2) receptive field를 사용하는 대신, 전체 network에서 stride 1의 3x3 사이즈를 사용한다. 3x3 convolutional layer 2개를 쌓는 것은 5x5 의 효과를 내고 3개를 쌓으면 7x7의 효과를 낸다. 이로써 decision function을 더욱 discriminative하게 만들고 parameter의 수를 줄일 수 있다.

1x1 convolutional layer의 사용으로 receptive field에 영향을 주지 않고 decision function의 비선형성을 증가시킬 수 있다. 
</br>

## 3. Classification Framework
### 3.1 Training
학습은 mini-batch gradient descent을 이용해 multinomial logistic regression 목적함수를 최적화하는 방식으로 이루어졌다. 배치 사이즈는 256, momentum은 0.9로 설정하였다. weight decay(5·0.0001), 그리고 첫 번째와 두 번째 fully connected layer에 dropout(p=0.5)를 추가하여 regularization 하였다. learning rate은 초기에 0.01로 설정하고 validation accuracy가 개선되지 않으면 1/10로 줄여나갔다. 총 3번의 learning rate 감소가 일어났으며 전체 74 epoch동안 학습을 진행하였다. Krizhevsky(2012)에 비해 더 깊은 네트워크와 많은 parameter에도 불구하고 더 빨리 수렴한 데에는 다음의 두 이유 때문인 것으로 추측된다.
  - 더 큰 depth와 더 작은 필터 사이즈에 의한 명백한 regularization
  - 특정 layer의 pre-initialization

깊으 architecture를 학습시킬 때, 앞 4개의 convolutional layer와 마지막 3개의 fully connected layer의 weight를 초기화하였다. pre-initialized 된 layer에 대해서는 learning rate를 감소시키지 않았다. 그 외의 random initialization은 평균이 0, 분산이 0.01인 정규분포로부터 이루어졌다. bias는 0으로 초기화하였다. 

224x224 사이즈의 입력 이미지를 얻기 위해 random crop을 수행하였다. augmentation으로 random horizontal flip과 random RGB color shift도 이루어졌다.

**Training Image Size.** 학습 이미지의 짧은 쪽 변의 길이를 S라고 하자.(여기서 S를 training scale 이라고 부른다) 모델의 입력으로는 원본 이미지를 224x224로 자른 이미지가 사용된다.

training scale S를 정하는 두 가지 방법을 고려한다. 첫 번째는 single-scale training으로, 고정된 값을 사용하는 것이다. 연구에서는 256과 384, 두 가지 값을 사용하였다. 먼저 S=256으로 네트워크를 학습시킨다. 그리고 S=384로 다시 학습시키는데 이때 S=256에서 학습된 weight로 초기화하여 속도를 높인다.

두 번째 방법은 multi-scale training으로, 각 학습 이미지에 대해 [S_min, S_max]에서 랜덤하게 선택된 S 값을 사용한다(논문에서는 S_min=256, S_max=512 사용). 학습 데이터셋에 대해 다양한 스케일을 이용하므로 augmentation으로 볼 수도 있다. 학습 속도 때문에, multi-scale 모델들은 S=384인 single-scale 사전학습 모델을 fine tuning하여 학습을 진행했다.


### 3.2 Testing
먼저 테스트 이미지의 짧은 변의 길이인 Q (test scale)로 rescale한다. 그리고 fully-convolutional net을 적용하여 그 결과로 class scoring map을 얻는다. 이를 공간적으로 평균 내어 (sum-pooled) 고정된 사이즈의 벡터를 얻는다. 또한 테스트 이미지를 horizontal flipping 하고 원본과 flipped 이미지의 결과를 평균내어 최종 결과로 사용한다.


</br>

### 3.3 Implementation Details
Multi-GPU 학습은 data parallelism을 이용하고, 각 학습 이미지 배치를 몇 개의 GPU에 나누어 각각 병렬적으로 처리한다. GPU 배치 gradient를 계산한 후, 평균 내어 전체 배치의 gradient를 계산한다. gradient 계산은 여러 GPU에서 동기적으로 일어나므로, 그 결과는 하나의 GPU에서 계산되었을 때와 동일하다. 

학습 속도를 높이는 다른 방법들도 제안되었지만, 본 연구진이 제안한 4-GPU 시스템은 하나의 GPU를 사용했을 때보다 3.75배 빠른 성능을 보여주었다. 4개의 NVIDIA Titan Black GPU를 사용했을 때, 하나의 네트워크 학습에 2~3주 소모하였다. 
</br>

## 4. Classification Experiments
### 4.1 Single Scale Evaluation
test 이미지 사이즈는 다음과 같다: 고정된 S에 대해서는 Q = S, S_min < S < S_max 인 S에 대해서는 Q = 0.5(S_min + S_max). 결과는 Table 3에 나타내었다. 

<p align="center"><img src="https://user-images.githubusercontent.com/86872735/158407223-83b5026a-748e-40a9-b9c1-c99d05b93adc.png"></p>

우선, 모델 A에 local response normalization을 추가하는 것(A-LRN)이 성능을 향상시키지 않는다는 것을 알았다. 따라서 더 깊은 모델인 B~E에는 local response normalization을 추가하지 않았다.

다음으로, 모델의 깊이를 늘릴수록 분류 오차가 감소함을 관찰했다. 특히, 동일한 깊이에도 불구하고, 3개의 1x1 conv. layer를 갖는 C가 네트워크 전체에서 1x1 conv. layer를 사용하는 D보다 성능이 안 좋았다. 이는 추가적인 non-linearity가 도움이 되긴 하지만, non-trivial receptive field에 conv. filter를 사용하여 공간적 context를 포착하는 것이 더 중요함을 의미한다. architecture의 오차율은 깊이가 19 layer에 달하면 포화하지만, 더 큰 데이터셋에 대해서는 더 깊은 모델이 효과적일 것이다. 또한 네트워크 B와, B의 3x3 conv. layer쌍을 하나의 5x5 conv. layer로 바꾼 네트워크를 비교해본 결과, B의 top-1 error rate이 7% 낮았다.

마지막으로, 테스트에서 single scale이 사용되었음에도 불구하고, 학습에서 S in [256, 512]의 scale jittering이 S = 256 또는 384의 고정된 경우보다 훨씬 좋은 성능을 내었다. 이는 scale jittering을 통한 augmentation이 multi-scale 이미지 통계를 포착하는데 도움을 준다는 것을 의미한다. 

### 4.2 Multi-Scale Evaluation

