> # Going Deeper with Convolutions (2014)
> ## Christian Szegedy et al.

</br>

## 1. Introduction
본 논문에서는 모델을 디자인할 때, 순전히 accuracy numbers (error rate나 정확도 얘기하는 듯?)에 집착하기보다는 메모리 사용량과 같은 알고리즘 효율성을 고려하였다.
대부분의 실험에서 연산량의 한계를 정해두었기 때문에 단순히 학문적인 관점 뿐만 아니라 현실에서도 합리적인 비용으로 사용할 수 있다.

본 논문에서 'deep' 이라는 용어는 두 가지 의미로 사용된다:
- 일반적인 의미 (신경망의 층이 깊어지는 것)
- Inception 모듈 형태의 조직의 새로운 레벨 ????
</br>
</br>

## 2. Related Work

## 3. Motivation and High Level Considerations
모델의 성능을 높이는 가장 간단한 방법은 모델의 depth와 width를 늘리는 것이다. 그러나 이는 overfitting과 연산량 증가라는 문제를 야기한다. 
해당 문제들을 해결하기 위한 근본적인 방법은 sparsity를 도입하고 심지어 fully-connected layer도 sparse한 것으로 대체하는 것이다. 
기존 연구에 의하면 데이터셋의 확률분포가 아주 크고 희박한 심층신경망으로 표현된다면,최적의 network topology는 이전 층의 활성화 함수들의 상관관계를 분석하고 
상관관계가 높은 뉴런들을 군집화하여 층층이 구성할 수 있다.
