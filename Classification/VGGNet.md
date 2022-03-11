> # Very Deep Convolutional Networks For Large-Scale Image Recognition (2014)
> ### Karen Simonyan, Andrew Zisserman
</br>

## 1. Introduction
CNN이 컴퓨터비전 분야에서 점점 더 사용되면서, 기존의 구조를 개선하여 성능을 높이려는 많은 노력들이 이루어졌다. 본 논문에서는 그 중 모델의 depth 문제를 다룬다. architecture의 다른 parameter는 고정하고, convolutional layer를 추가하여 network의 깊이만 조금씩 늘려간다. 이는 모든 layer에서 3x3 크기의 매우 작은 filter를 사용하기 때문에 가능하다.

결과적으로, ILSVRC에서 가장 좋은 성능을 냈을 뿐만 아니라 다른 dataset에도 적용 가능한 매우 정확한 CNN architecture를 얻을 수 있었다. 

## 2. ConvNet Configurations
