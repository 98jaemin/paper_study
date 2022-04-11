> # Rethinking the Inception Architecture for Computer Vision
> ### Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna (2015)


## 1. Introduction
AlexNet은 2012 ImageNet 대회에서 우승한 이후, object-detection, segmentation, human pose estimation, video classification, object tracking, superresolution과 같은 다양한 컴퓨터비전 분야에 성공적으로 적용되어 왔다. 그 이후 VGGNet이나 GoogLeNet과 같은 network들도 등장하여 높은 성능을 보여주었다. 

비록 VGGNet은 구조적 단순함이라는 강력한 특징을 갖고 있지만 계산 비용이 너무 높다. 반면 GoogLeNet의 Inception architecture는 메모리와 계산 비용을 제한하고도 좋은 성능을 보인다. GoogLeNet은 AlexNet보다 12배 적은 parameter를 사용하는데 VGGNet은 AlexNet보다 3배나 많은 parameter를 사용한다.

