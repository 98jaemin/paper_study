> # ImageNet Classification with Deep Convolutional Neural Networks (2012)
> ### Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
</br>

## 1. Introduction
본 논문의 contribution은 다음과 같다:
- 2010년과 2012년 ImageNet 대회의 데이터셋을 이용하여 역대급으로 큰 규모의 CNN 모델 학습
- 여러 새로운 학습 기법을 사용 (3장에서 설명)
- overfitting을 막기 위한 여러 기법을 사용 (4장에서 설명)

전체 모델은 5개의 convolutional layer와 3개의 fully-connected layer로 구성되어 있으며 하나의 convolutional layer라도 제거하면 성능이 급격히 저하된다.

신경망의 사이즈는 GPU의 성능과 training time의 한계에 봉착했다. 학습은 GTX 580 3GB GPU 2개를 이용하여 6일 정도 소요되었으며 더 좋은 GPU와 더 큰 데이터셋이 등장하면 성능이 향상될 것으로 기대한다.
</br>
</br>

## 2. Dataset
ImageNet은 약 22,000가지의 카테고리를 갖는 150만 개 이상의 이미지로 구성된 데이터셋이다.
ILSVRC에서는 ImageNet의 1,000가지 카테고리 이미지를 각 1,000장씩을 추출하여, training 120만 장, validation 5만 장, testing 15만 장을 사용한다.
ILSVRC-2010만이 test set 레이블을 이용할 수 있으므로 본 논문에서는 해당 대회의 이미지를 사용한다.</br>

ImageNet은 다양한 사이즈의 이미지로 이루어져있다. 그러나 AlexNet은 256x256 사이즈만 입력 받으므로 사이즈 조정이 필요하다.
먼저 이미지의 짧은 쪽을 256으로 조절한 후 이미지의 중앙 부분을 256x256 사이즈로 잘라내어 사용한다.
전처리로는 각 픽셀에서 전체 training set의 픽셀값의 평균을 빼는 작업을 수행하였다.
</br>
</br>

## 3. Architecture
### 3.1 ReLU Nonlinearity
기존에 사용하던 activation 함수는 tanh 또는 sigmoid 함수였다. 학습 시간 면에서 이들보다 max(0, x)가 훨씬 빠르다. 이 함수를 사용한 뉴런을 ReLU(Rectified Linear Units)라고 부르기로 했다. 아래 그림은 4개의 layer를 갖는 CNN 모델이 CIFAR-10 데이터셋을 학습하여 25%의 training error에 도달하는 데에 걸리는 epoch을 나타낸다. 실선은 ReLU, 점선은 tanh을 이용한 모델이다.

<p align="center"><img src="https://user-images.githubusercontent.com/86872735/155063648-90cbf655-65ca-483a-90f0-568672d9e77a.png" width="35%"></p>


본 논문이 기존 activation 함수의 대안을 제안한 최초의 사례는 아니다. Jarrett은 |tanh(x)| 함수를 이용하였고 좋은 성능을 냈다. 하지만 overfitting 방지와 빠른 학습의 면에서 ReLU가 더 효과적이다.


### 3.2 Training on Multiple GPUs
한 개의 GTX 580 GPU의 메모리 한계 때문에 본 연구에서는 2개의 GPU를 병렬로 사용하였다. 

"여기서 추가적인 기법이 있는데, 데이터를 두 개의 GPU로 나누어 학습시키다가 하나의 layer에서만 GPU를 통합시키는 것입니다. 논문에서는 3번째 Conv layer에서만 GPU를 통합시킨다고 말합니다. 이를 통해 계산량의 허용가능한 부분까지 통신량을 정확하게 조정할 수 있다고 나와있습니다"


### 3.3 Local Response Normalization
일반적으로 ReLU는 전통적인 activation 함수와 달리 saturating 되지 않으므로 굳이 normalization을 추가하지 않아도 된다. 그러나 본 연구에서는 다음과 같은 normalization을 사용했을 때 일반화 성능이 향상되는 것을 확인했다. 
<p align='center'><img src='https://user-images.githubusercontent.com/86872735/155460331-afb7e938-c5c3-481b-8e5b-d215b62a35c6.png' width='50%'></p>



### 3.4 Overlapping Pooling
pooling layer는 동일한 kernel map 내의 이웃한 뉴런들의 정보를 요약하는 역할을 한다.
전통적으로 pooling layer는 overlap 되지 않게(stride를 s, kernel size를 z라고 하면 s = z) 적용해 왔으나 본 논문에서는 overlap 되도록 (s < z) s = 2, z = 3 으로 설정하였다. 그 결과 성능이 향상되었으며 overfitting을 약간 방지하는 효과도 확인하였다.


### 3.5 Overall Architecture
<p align='center'><img src="https://user-images.githubusercontent.com/86872735/155250730-f199b8ee-07ed-445e-a3dc-8dfe30c80b9f.png"></p>

네트워크는 5개의 convolutional layer와 3개의 fully-connected layer로 구성되며 마지막 fully-connected layer의 출력은 softmax 함수를 통해 1000개의 label 중 하나로 결정된다.

* 1st Convolutional Layer
  * input : 224x224x3
  * kernel size : 11x11x3
  * \# of kernels : 96
  * stride : 4
* Response Normalization + Max Pooling
* 2nd Convolutional Layer
  * kernel size : 5x5x48
  * \# of kernels : 256
* Response Normalization + Max Pooling
* 3rd Convolutional Layer
  * kernel size : 3x3x256
  * \# of kernels : 384
* 4th Convolutional Layer
  * kernel size : 3x3x192
  * \# of kernels : 384
* 5th Convolutional Layer
  * kernel size : 3x3x192
  * \# of kernels : 256
* Max Pooling (설명은 없으나 그림에 쓰여있음)
* Fully-Connected Layer
  * \# of neurons : 4096
</br>
</br>

## 4. Reducing Overfitting
### 4.1 Data Augmentation
overfitting을 줄이는 가장 간단한 방법은 label-preserving transformation을 이용하여 이미지를 늘리는 것이다. 본 논문에서는 두 가지의 augmentation 방법을 사용하였다.

첫 번째는 이미지의 이동과 수평 반전이다. 256x256 사이즈의 이미지에서 224x224 사이즈의 patch를 추출하여 학습에 이용하였다. 네 부분의 코너와 중앙 부분 중 하나를 선택하고 수평 반전 또한 적용되므로 하나의 이미지에 대해 10가지 경우 중 하나가 선택되는 것이다. 

두 번째는 이미지의 RGB 값의 강도를 조절하는 것이다. RGB 픽셀 값에 주성분 분석을 실시한 뒤 고윳값에 비례하여 주성분의 배수를 이미지에 더했다 ??????????

### 4.2 Dropout
여러 다른 모델의 예측을 결합하는 것은 test error를 줄이는 좋은 방법이지만 한번 학습하는데 며칠씩 걸리는 매우 큰 모델에는 적용하기 어렵다. 하지만 효율적으로 그러한 효과를 내는 방법이 바로 "dropout"이다. hidden layer의 각 뉴런은 0.5의 확률로 "drop out" 되며 제거된 뉴런은 예측에 참여하지 않는다. 따라서 매 입력마다 모델은 서로 다른 구조를 가지며 뉴런들은 더 robust한 특징들을 학습하게 된다.
</br>
</br>

## 5. Details of learning
* optimizer : Stochastic Gradient Descent
* momentum : 0.9
* weight decay : 0.0005
* batch size : 128
* weight 초기화 : N(0, 0.01^2)
* bias 초기화
 * 2, 4, 5 번째 convolutional layer, fully-connected layer : 1
 * 1, 3 번째 convolutional layer : 0

