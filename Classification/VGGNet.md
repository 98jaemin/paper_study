> # Very Deep Convolutional Networks For Large-Scale Image Recognition (2014)
> ### Karen Simonyan, Andrew Zisserman
</br>

## 1. Introduction
CNN이 컴퓨터비전 분야에서 점점 더 사용되면서, 기존의 구조를 개선하여 성능을 높이려는 많은 노력들이 이루어졌다. 본 논문에서는 그 중 모델의 depth 문제를 다룬다. architecture의 다른 parameter는 고정하고, convolutional layer를 추가하여 network의 깊이만 조금씩 늘려간다. 이는 모든 layer에서 3x3 크기의 매우 작은 filter를 사용하기 때문에 가능하다.

결과적으로, ILSVRC에서 가장 좋은 성능을 냈을 뿐만 아니라 다른 dataset에도 적용 가능한 매우 정확한 CNN architecture를 얻을 수 있었다. 

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

## 3. Classification Framework
### 3.1 Training
학습은 mini-batch gradient descent을 이용해 multinomial logistic regression 목적함수를 최적화하는 방식으로 이루어졌다. 배치 사이즈는 256, momentum은 0.9로 설정하였다. weight decay(5·0.0001), 그리고 첫 번째와 두 번째 fully connected layer에 dropout(p=0.5)를 추가하여 regularization 하였다. learning rate은 초기에 0.01로 설정하고 validation accuracy가 개선되지 않으면 1/10로 줄여나갔다. 총 3번의 learning rate 감소가 일어났으며 전체 74 epoch동안 학습을 진행하였다. Krizhevsky(2012)에 비해 더 깊은 네트워크와 많은 parameter에도 불구하고 더 빨리 수렴한 데에는 다음의 두 이유 때문인 것으로 추측된다.
  - 더 큰 depth와 더 작은 필터 사이즈에 의한 명백한 regularization
  - 특정 layer의 pre-initialization

깊으 architecture를 학습시킬 때, 앞 4개의 convolutional layer와 마지막 3개의 fully connected layer의 weight를 초기화하였다. pre-initialized 된 layer에 대해서는 learning rate를 감소시키지 않았다. 그 외의 random initialization은 평균이 0, 분산이 0.01인 정규분포로부터 이루어졌다. bias는 0으로 초기화하였다. 

224x224 사이즈의 입력 이미지를 얻기 위해 random crop을 수행하였다. augmentation으로 random horizontal flip과 random RGB color shift도 이루어졌다.

**Training Image Size** ??

### 3.2 Testing




