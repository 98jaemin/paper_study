> # Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (2015)
> ### Sergey Ioffe, Christian Szegedy
</br>

## 1. Introduction
딥러닝은 vision과 speech 등 다양한 분야에서 급속도로 발전하고 있다. 확률적 경사하강법(SGD)는 심층신경망을 학습시키는 효율적인 방식임이 증명되었고, momentum이나 Adagrad와 같은 변형들도 사용되고 있다. SGD의 각 학습 단계에서 크기가 m인 mini-batch를 고려한다. mini-batch는 parameter에 대한 loss function의 gradient를 근사하기 위해 사용된다.

<p align="center"><img src="https://user-images.githubusercontent.com/86872735/159493642-794bb5cc-39e9-42b8-9234-7c076a42425c.png" width="15%"></p>

한번에 하나의 데이터를 사용하는 것과 비교하여 mini-batch를 사용하는 것은 몇 가지 이점이 있다.
- mini-batch loss의 gradient는 전체 training set의 gradient의 추정값이며 이는 batch size가 커질수록 개선된다
- 각 데이터에 대해 m번 연산하는 것보다 batch를 통해 연산하는 것이 훨씬 효율적이다. (parallelism)

각 layer의 입력은 모든 이전 layer의 parameter에 영향을 받게 되며, 따라서 network가 깊어질수록 작은 변화도 증폭되게 된다.

layer의 입력의 분포의 변화는 layer가 계속해서 새로운 분포에 적응해야 한다는 문제를 야기한다. network가 다음과 같은 연산을 한다고 하자. F_1과 F_2는 임의의 변환이고 Theta_1과 Theta_2는 loss를 최소화하기 위해 학습되는 parameter이다.
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/159498055-8b797b0f-7fb9-4983-be10-ecddcb602b11.png" width="22%"></p>

Theta_2에 대한 학습은 x=F_1(u, Theta_1)일 때 다음과 같다.
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/159498647-3822f745-41c3-438f-a8be-896bc46b4f76.png" width="15%"></p>

gradient descent 단계에서 연산을 할 때, 이는 network F_2에 입력 x를 받아들이는 것과 동등하다. 입력 분포의 특성은 sub-network에도 적용된다. x의 분포가 고정된다면 Theta_2는 매번 변화에 맞추어 재조정될 필요가 없다.

입력의 고정된 분포는 sub-network 바깥에도 그 이점이 있다. sigmoid 활성화 함수를 생각해보자. 0 근처가 아닌 모든 x에 대해 기울기 소실 문제가 발생하고 학습이 느리게 이루어진다. 이러한 포화 문제를 해결하기 위해 일반적으로 초기화를 조정하거나 ReLU, 또는 더 작은 learning rate을 사용한다. 하지만 비선형적 입력의 분포가 학습 시 더 안정적으로 유지된다면 optimizer는 기울기 포화에 빠질 가능성이 줄어들고 학습은 가속될 것이다.

연구진은 학습 과정에서 심층신경망의 내부 노드의 분포가 변하는 것을 ***Internal Covariate Shift***라고 정의하였다. 이것을 제거하는 것은 학습이 빨라지도록 한다. 연구진은 internal covariate shift를 제거하고 학습 속도를 급격히 빠르게 만드는 ***Batch Normalization***이라는 새로운 메커니즘을 제안한다. 이것은 normalization 단계를 통해 layer 입력의 평균과 표준편차를 고정함으로써 이루어진다. Batch Normalization은 network 전체의 gradient 흐름에도 이점이 있는데, parameter의 scale이나 초깃값에 대한 gradient의 의존도를 감소시킨다. 이로 인해 발산의 위험성 없이 더 큰 learning rate을 사용할 수 있게 된다. 게다가 Batch Normalization은 모델을 규제하여 Dropout의 필요성을 감소시킨다. 마지막으로 기울기의 포화도 예방한다.

§4.2에서 현재 가장 뛰어난 ImageNet 분류 network에 Batch Normalization을 적용하였고 그 결과 단 7%의 학습 단계만에 동등한 성능을 낼 수 있었다. Batch Normalization을 사용한 network를 앙상블하여 알려진 최고 수준의 top-5 error rate을 달성하였다.
</br>
</br>

## 2. Towards Reducing Internal Covariate Shift
입력이 whitened(평균 0, 분산이 1을 갖고 상관성이 없도록 선형 변환하는 것을 whitening 이라고 표현) 되었을 때 network의 학습이 빨라진다는 것은 오래전부터 알려져 있었다. 각 layer의 입력을 whitening함으로써 분포를 고정하고 internal covariate shift의 나쁜 효과를 제거하는데에 한 발 다가가고자 한다.

연구진은 매 학습 단계 또는 특정한 간격으로 network를 직접 수정하거나 parameter를 변경함으로써 whitening activation을 고려하였다. 그러나 이러한 수정들이 optimization 단계에 배치된다면, gradient descent 단계에서 parameter는 normalization의 업데이트를 요구하는 방향으로 개선될 것이고, 이는 gradient 단계의 효과를 감소시킨다.

위의 접근법의 문제는 gradient descent optimization이 normalization이 일어난다는 점을 고려하지 않는다는 것이다. 이 문제를 해결하기 위해, 임의의 parameter 값에 대해 network가 항상 이상적인 활성화를 출력하도록 보장하고자 한다. 그렇게 함으로써 loss의 gradient가 normalization과 모델 parameter의 의존성을 설명할 수 있게 된다. 

x를 layer 입력 벡터, X를 학습데이터셋의 x의 집합이라고 할 때 normalization은 Norm(x, X)로 쓸 수 있다. 즉 입력 x 뿐만 아니라 parameter에 의존하는 모든 X에 의존하게 된다. 이와 같은 framework에서 layer 입력을 whitening 하는 것은 많은 계산량을 요구한다. 따라서 연구진은 미분이 가능하며, 매 parameter 업데이트마다 전체 학습 데이터셋의 분석을 필요로 하지 않는 새로운 normalization 방법을 탐색하였다. 

기존의 몇몇 접근법들은 하나의 학습 샘플만을 이용하거나, 이미지의 경우 동일한 위치의 여러 샘플을 이용하였다. 하지만 이러한 방법은 activation의 절대적 스케일을 버림으로써 표현력을 바꾸게 된다. 연구진은 전체 학습 데이터의 통계와 관련하여 학습 샘플을 normalization함으로써 network의 정보들을 보존하고자 한다.
</br>
</br>

## 3. Normalization via Mini-Batch Statistics
먼저, layer 입력과 출력의 feature들을 함께 whitening 하는 대신, 각 scalar feature를 독립적으로 normalize 한다. d차원 입력 x에 대해 각 차원을 다음과 같이 normalize하며 기댓값과 분산은 전체 학습 데이터셋을 통해 계산된다.
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/160398672-a4ce55d3-9c08-40ed-af99-f01cf2a2b011.png" width="20%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/160398701-36a091a4-d719-4745-b60e-d3f7941d0403.png" width="20%"></p>

단순히 각 layer의 입력을 normalizing하는 것은 해당 layer의 표현력을 변화시킬 수 있다. 이 문제를 해결하기 위해, network에 추가된 변환이 identity transform을 대신할 수 있도록 한다. 그렇게 하기 위해 연구진은 normalize된 값을 scale 및 shift 하는 parameter, gamma와 beta를 도입한다.
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/160400594-03256a41-e3dc-4273-a922-81ffbb024005.png" width="20%"></p>

이 parameter들은 원래의 모델 parameter와 함께 학습되며 network의 표현력을 저장한다. 사실 gamma^(k) = sqrt(Var[x^(k)]), beta^(k) = E[x^(k)]로 설정하면 기존 activation을 복구할 수 있다.

우리는 mini-batch SGD를 사용하므로, mini-batch가 각 activation의 평균 및 분산의 추정량을 만들어낸다고 하자. 그러면 normalization에 사용한 통계량이 gradient backpropagation에 완전히 참여할 수 있다. mini-batch의 사용은 결합공분산이 아닌 차원별 분산의 계산 덕에 가능함을 주목한다.

크기 m의 mini-batch B를 생각하자. 어떤 activation x^(k)에 대해 k를 떼어두면 mini-batch에는 m개의 activation이 존재한다. 정규화된 값을 x_hat, 변환된 값을 y라고 하면 Batch Normalizing Transform은 다음과 같다. 
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/160408202-dd1b61ca-6a95-4739-8591-283a522b6299.png" width="50%"></p>

Batch Normalization은 학습 샘플 x와 mini-batch 내의 다른 학습 샘플들에 의존한다. normalized activation x^k_hat은 변환의 내부에 있지만 그 존재는 매우 중요하다. 각 x^k_hat은 y = gamma * x + beta로 구성된 sub-network의 입력으로 볼 수 있으며 그 다음에는 원래의 network의 다른 처리가 뒤따른다. 따라서 그 입력은 모두 고정된 평균과 분산을 갖게 되며, 비록 결합분포는 학습 과정 중 변할 수 있지만 우리는 이로 인해 학습이 가속화 될 것이라고 예상한다. 

학습 동안 loss의 gradient의 backpropagation을 진행해야 하고 BN transform의 parameter에 대한 gradient를 계산해야 한다. 
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/160576002-ec208b40-cf96-4e68-b201-f9cb9a509e14.png"></p>

따라서 BN transform은 normalized activation을 network로 도입하는 normalized transform이다. 이것은 training 단계에서 layer가 더 적은 internal covariate shift를 나타내는 입력 분포를 학습하도록 보장하므로 training이 가속화된다. 게다가 학습되어 이런 normalized activation에 적용되는 affine tranform은 BN transform이 identity transform을 나타내도록 하며 network capacity를 보존한다.

### 3.1 Training and Inference with Batch Normalized Networks
Batch Normalization을 사용하는 모델은 batch size > 1인 mini-batch를 이용한 최적화 알고리즘으로 학습한다. mini-batch에 의존하는 activation의 normalization은 학습을 효율적으로 만들지만 추론 단계에서는 필요하지도, 바람직하지도 않다; 우리는 출력이 오직 입력에만 의존하기를 원한다. 따라서 network가 학습되고 나면, 우리는 mini-batch가 아닌 모집단을 이용하여 다음과 같이 normalization 한다.
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/160795341-a3a5b60c-31bf-4842-91f4-5c751976845c.png" width="20%"></p>
  
추론 단계에서는 평균과 분산이 고정되므로, normalization은 단순한 선형변환이 된다. scaling과 shift를 위한 gamma와 beta를 이용하여 BN을 대신할 single linear transform으로 구성될 수도 있다. 
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/160796265-4f7a1a9f-a2e7-4310-96ff-3c2675c0df03.png" width="50%"></p>

### 3.2 Batch-Normalized Convolutional Networks
이 절에서는 element-wise한 비선형성의 affine transformation으로 구성된 변환에 대해 이야기한다 - W와 b는 모델 parameter, g()는 sigmoid와 같은 비선형함수일 때의 z = g(W·u + b). 

우리는 u를 normalize할 수도 있지만 u는 다른 비선형 함수의 출력이기 때문에 그 분포는 학습 동안 달라지며 u의 first and second moment를 제한하는 것은 covariate shift를 제거하지 못한다. 반면, W·u + b는 더 대칭적이고, non-sparse한 분포를 가지므로 이것을 normalize하면 안정적인 분포의 activation을 얻게 된다.

우리는 W·u + b를 normalize하기 때문에, bias b의 효과는 이후의 mean subtraction에 의해 상쇄되며 따라서 b는 무시할 수 있다. 그러므로 z = g(W·u + b)를 x = W·u의 각 차원에 독립적으로 BN transform이 적용된 z = g(BN(W·u))로 대체한다.

convolutional layer에 대해, 우리는 추가적으로 normalization이 convolutional property를 따르기를 바란다 - 그렇게 되면 동일한 feature map의 서로 다른 위치의 원소들이 같은 방식으로 normalize 된다. 이를 위해, 우리는 mini-batch의 모든 위치의 activation을 공동으로 normalize 한다. Alg. 1에서 feature map의 모든 값의 집합을 B라고 놓는다. 그리고 크기 m의 mini-batch와 pxq feature map에 대해 효율적인 mini-batch 크기로 m' = |B| = m·p·q를 사용한다. parameter 쌍 gamma와 beta는 매 activation이 아닌 매 feature map마다 학습한다. Alg 2.도 비슷하게 수정하여 inference 동안 BN transform은 주어진 feature map의 각 activation에 동일한 linear transformation을 적용한다.

### 3.3 Batch Normalization enables higher learning rates
Batch Normalization은 너무 큰 learning rate을 사용하여 발생하는 gradient explosion, vanishing 등의 문제를 해결하는데 도움을 준다. activation을 normalizing함으로써 parameter의 작은 변화가 크고 최적이 아닌 변화로 증폭되는 것을 방지한다.

또한 Batch Normalization은 학습이 parameter scale에 대해 더 resilient하게 만든다. 일반적으로 큰 learning rate은 layer parameter의 scale을 키우고, 따라서 backpropagation 동안 gradient를 증폭하여 explosion을 야기한다. 그러나 Batch Normalization을 사용하면 backpropagation 동안 parameter의 scale에 영향을 받지 않는다. 

게다가 우리는 Batch Normalization이 layer Jacobians의 특잇값이 1에 가까워지도록 한다고 추측하는데 이는 학습에 도움이 되는 것으로 알려져있다. 

### 3.4 Batch Normalization regularizes the model
Batch Normalization을 이용한 학습 과정에서, 학습 샘플이 mini-batch 내의 다른 샘플과 결합?(conjunction)을 이루는 것을 보았고, 학습 network는 더 이상 주어진 학습 샘플에 대해 결정론적인 값을 도출하지 않았다. 우리는 이 효과가 network의 일반화에 도움이 된다는 것을 알아냈다. overfitting을 줄이기 위해 Dropout이 주로 사용되는데, batch-normalized network에서는 Dropout을 제거하거나 그 강도를 줄여도 됨을 확인했다.

## 4. Experiments
### 4.1 Activations over time
internal covariate shift의 영향과 Batch Normalization의 능력을 검증하기 위해 MNIST 데이터셋을 사용하였고, 100개의 activation을 갖는 3개의 fully-connected layer로 구성된 간단한 모델을 이용하였다. 각 hidden layer는 sigmoid를 이용하였고 weight는 Gaussian을 이용해 초기화하였다. EPOCH은 50,000, batch size는 60으로 학습하였다. 각 hidden layer마다 Batch Normalization을 추가하였다. 
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/160974703-5c898188-ec20-4b80-baf8-d7870f111783.png"></p>

Figure 1(a)에서 볼 수 있듯, batch-normalized network가 더 빠르고 좋은 성능을 내었다. 그 이유를 찾기 위해 sigmoid의 input을 조사하였다. Figure 1(b, c)에 마지막 hidden layer의 하나의 activation의 분포가 어떻게 발전하는지 나타냈다. 일반 network(b)에서는, 학습의 진행에 따라 분포가 상당히 변화하며 이는 그 다음 layer에서의 학습을 복잡하게 한다. 반면 batch-normalized network(c)에서의 분포는 훨씬 안정적이고 이는 학습에 도움이 된다.

## 4.2 ImageNet Classification
우리는 Inception Network의 새로운 변형에 Batch Normalization을 적용하였다. 기존과의 가장 큰 차이는 5x5 convolutional layer 대신 두 개의 연속적인 3x3 convolution을 사용한다는 점이다. network는 13,600,000개의 parameter를 가지며 모든 convolution layer는 ReLU를 사용하며, softmax layer를 제외한 fully-connected layer는 없다. 이 모델을 이하 Inception이라 칭한다. 이 Inception 모델을 mini-batch size 32의 SGD로 학습시켰다. 

실험에서 우리는 Batch Normalization을 이용한 Inception의 약간의 변형들을 평가하였다. 모든 경우 Batch Normalization은 각 non-linearity의 input에 적용되었다. 

### 4.2.1 Accelerating BN Networks
단순히 Batch Normalization을 추가하기만 하는 것은 그 장점을 완전히 이용하지 못한다. 따라서 우리는 network와 학습 parameter를 약간 변경하였다:
- Increase learning rate
- Remove Dropout
- Reduce L2 weight regularization 
- Accelerate learning rate decay : 우리의 network가 Inception보다 빠르게 학습되었기 때문에, learning rate을 6배 빠르게 줄여나갔다.
- Remove Local Response Normalization
- Shuffle training examples more thoroughly
- Reduce the photometric distortions : batch-normalized network의 학습이 빠르고 각 학습 샘플을 덜 관찰하므로, 우리는 학습기가 왜곡이 되지 않은 실제 이미지에 더 집중하도록 한다.

### 4.2.2 Single-Network Classification
LSVRC 2012 데이터를 이용하여 아래의 network들을 평가
- Inception : §4.2에서 소개한 network, 초기 learning rate은 0.0015
- BN-Baseline : Inception가 동일하되 각 nonlinearity 앞에 Batch Normalization 추가
- BN-x5 : Inception + Batch Normalization + §4.2.1의 수정사항, 초기 learning rate은 0.0075
- BN-x30 : BN-x5와 동일하되 초기 learning rate은 0.045
- BN-x5-Sigmoid : BN-x5와 동일하되, ReLU 대신 Sigmoid
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/161465885-5f8808f8-c4ad-4ba6-9213-5bbb86cc8418.png" width="60%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/86872735/161466044-2e2b162d-2bfb-4e61-a583-cf50d439a365.png" width="60%"></p>

Figure 2에 각 network의 validation accuracy를 나타내었고 Inception은 31·10^6 step만에 72.2%를 달성하였다.
Figure 3은 각 network가 72.2%를 달성하는데 걸린 step의 수와 최대 정확도를 나타내었다.
Batch Normalization과 §4.2.1에서 소개한 수정사항을 적용하면 훨씬 빠르게 높은 정확도를 얻을 수 있음을 확인하였다. 또한 Batch Normalization은 Sigmoid의 학습의 어려움을 다소 해결할 수 있었다. Batch Normalization이 없으면, sigmoid를 사용한 Inception의 정확도는 0.1% 미만이다.

### 4.2.3 Ensemble Classification
6개의 network를 이용하여 앙상블하였다. 각각은 BN-x30을 기반으로 하였으며, convolutional layer의 weight 초깃값을 증가시키고 dropout을 추가하는 등 몇 가지 사항만 변경하였다. 각 network는 대략 6·10^6 step 이후 최대의 정확도를 얻었으며 앙상블 예측을 위해 각 class 확률값을 평균내었다. 그 결과 공식적인 최고 성능을 앞서는 결과를 얻을 수 있었다.
</br>
</br>

## 5. Conclusion
Batch Normalization은 학습을 복잡하게 만드는 covariate shift를 제거하면 학습에 도움이 된다는 것을 전제로 하여 시작했다. Batch Normalization은 어떤 최적화 알고리즘과도 적절히 사용될 수 있다. 그렇게 하기 위해 normalization을 각 mini-batch에 대해 진행하고, backpropagation에서 각 normalizatino parameter에 대해 gradient를 계산한다. Batch Normalization은 각 activation에 대해 단 2개의 parameter만을 추가하면서도 network의 표현력을 보존한다. 

단순히 현재 state-of-the-art model에 Batch Normalization을 추가하는 것만으로도 학습의 속도를 높이며, learning rate를 키우고 dropout을 제거하는 등 몇 가지를 수정하면 훨씬 적은 step만에 현재의 성능에 도달한다. 게다가 Batch Normalization을 이용한 여러 모델을 결합하면 현재의 sota 모델을 상당한 격차로 앞서게 된다. 









