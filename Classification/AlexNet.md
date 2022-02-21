> # ImageNet Classification with Deep Convolutional Neural Networks (2012)
> ### Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
</br>

## 1. Introduction

## 2. Dataset
ImageNet은 약 22,000가지의 카테고리를 갖는 150만 개 이상의 이미지로 구성된 데이터셋이다.
ILSVRC에서는 ImageNet의 1,000가지 카테고리 이미지를 각 1,000장씩을 추출하여, training 120만 장, validation 5만 장, testing 15만 장을 사용한다.
ILSVRC-2010만이 test set 레이블을 이용할 수 있으므로 본 논문에서는 해당 대회의 이미지를 사용한다.</br>

ImageNet은 다양한 사이즈의 이미지로 이루어져있다. 그러나 AlexNet은 256x256 사이즈만 입력 받으므로 사이즈 조정이 필요하다.
먼저 이미지의 짧은 쪽을 256으로 조절한 후 이미지의 중앙 부분을 256x256 사이즈로 잘라내어 사용한다.
전처리로는 각 픽셀에서 전체 training set의 픽셀값의 평균을 빼는 작업을 수행하였다. (<- 확인 필요)
