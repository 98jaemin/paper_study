> # Rethinking the Inception Architecture for Computer Vision
> ### Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna (2015)


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
