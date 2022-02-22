> # ImageNet Classification with Deep Convolutional Neural Networks (2012)
> ### Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
</br>

## 1. Introduction
수백만 장의 이미지를 학습하기 위해서는 더 큰 model capacity를 필요로 한다.
CNN 모델은 비슷한 사이즈의 순방향 신경망보다 훨씬 적은 수의 파라미터를 사용하지만 성능은 거의 동일(slightly worse)하다.

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
기존에 사용하던 activation 함수는 tanh 또는 sigmoid 함수였다. 학습 시간 면에서 이들보다 max(0, x)가 훨씬 빠르다. 이 함수를 사용한 뉴런을 ReLU(Rectified Linear Units)라고 부르기로 했다. Fig 1.은 4개의 layer를 갖는 CNN 모델이 CIFAR-10 데이터셋을 학습하여 25%의 training error에 도달하는 데에 걸리는 epoch을 나타낸다. 실선은 ReLU, 점선은 tanh을 이용한 모델이다.


<p align="center"><img src="https://user-images.githubusercontent.com/86872735/155063648-90cbf655-65ca-483a-90f0-568672d9e77a.png" align="center"></p>
<p align="center">Fig 1. 학습속도 비교 </p>


본 논문이 기존 activation 함수의 대안을 제안한 최초의 사례는 아니다. Jarrett은 |tanh(x)| 함수를 이용하였고 좋은 성능을 냈다. 하지만 overfitting 방지와 빠른 학습의 면에서 ReLU가 더 효과적이다.


### 3.2 Training on Multiple GPUs
한 개의 GTX 580 GPU의 메모리 한계 때문에 본 연구에서는 2개의 GPU를 병렬로 사용하였다. 

"여기서 추가적인 기법이 있는데, 데이터를 두 개의 GPU로 나누어 학습시키다가 하나의 layer에서만 GPU를 통합시키는 것입니다. 논문에서는 3번째 Conv layer에서만 GPU를 통합시킨다고 말합니다. 이를 통해 계산량의 허용가능한 부분까지 통신량을 정확하게 조정할 수 있다고 나와있습니다"




