---
header: Human-level control through deep reinforcement learning
mathjax: True
---
이 논문은 2013년의 [“Playing Atari with Deep Reinforcement Learning”](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) 연구 내용을 좀더 정교하게 다듬고 새로운 내용 몇가지를 추가해 2015년 5월에 [Nature 에 게재된 논문](https://www.nature.com/articles/nature14236)입니다. 따라서 본 리뷰에서는 2013년 연구와 중복되는 내용은 대부분 배제하고 새로운 내용 위주로 정리하였습니다. 2013년 연구에 대한 리뷰는 [직전의 포스팅](https://medipixel.github.io/Playing_Atari_with_Deep_Reinforcement_Learning)을 참고하시면 됩니다. 이 논문에서 새로 추가된 내용의 핵심은 target Q-network, Error clipping 등이며, 2013년보다 더 깊은 네트워크와 학습 시간을 이용하여 더 많은 게임에 DQN 알고리즘을 적용하였습니다.

<!--break-->

***
### 0. Novelty
2013년 연구와 비교할때 대표적인 novelty는 target Q-network 입니다. 

강화학습은 action-value function (Q function) 을 표현하기 위해 신경망과 같은 nonlinear function approximator를 사용할 경우 불안정하거나 발산할 수 있는 것으로 알려져 왔습니다. 그 이유로는 1) 관찰하는 시퀀스 간의 상관관계, 2) Q에 대한 작은 업데이트가 policy를 크게 변화시키고 data distribution을 변화시켜 버림, 3) action-value (Q) 와 target value 간의 상관관계 등이 있습니다. 

이에 대한 해법은 두가지입니다. 첫번째는 experience replay로, 이에 대한 자세한 설명은 2013년 논문 리뷰를 참고하기 바랍니다. 두번째 해법은 별도의 target Q-network 를 준비함으로써 action-value (Q) 와 target value 간의 상관관계의 문제를 해결하는 것입니다. 별도의 target Q-network는 다음과 같이 진행합니다. 

C번째 업데이트 시마다 network Q를 복제하여 target network Q 을 만듭니다. Q 을 이용하여 그 다음의 C번의 업데이트를 위한 Q-learning target인 yi 를 생성합니다. 이런 과정이 없다면 Q(st, at) 를 증가하도록 업데이트하면 Q(st+1, at) 역시 증가하기 때문에 policy의 oscillation 또는 divergence를 유발하게 되기 때문입니다. 

다시 말해 Q value가 Q value 자신에 의존하여 업데이트를 하는 식의 재귀적 (recursive) 방식이기 때문입니다. 쉽게 말해서 target에 접근하도록 업데이트를 해줬는데 정작 target은 다시 도망가는 모양새인 것입니다. 

기존의 파라미터를 이용하여 타겟을 생성함으로써 Q에 대한 업데이트가 이루어지는 시간과 이 업데이트가 타겟 yj에 영향을 미치는 시간 사이에 지연을 줌으로써 oscillation이나 divergence가 생길 가능성을 낮춰주는 것입니다. 


### 1. Methods 
#### (1) Preprocessing

아타리 에뮬레이터는 그 성능의 한계로 인해 일부 오브젝트들은 홀수 프레임에만, 다른 오브젝트들은 짝수 프레임에만 존재하게 설계가 되어 있습니다. 따라서 본 연구에서는 현재의 프레임과 직전의 프레임에 걸쳐 각 픽셀 컬러 값의 최대 값을 취하는 방식으로 하나의 프레임을 인코딩하였습니다. 색상을 제거하여 gray color로 변환하고 84 x 84 의 크기로 down-sampling 해 주었습니다.

Q-function에 입력은 m개의 최근 프레임을 스택하여 이용하였습니다. m의 값으로는 4를 이용하였으나 3이나 5를 써도 결과는 동일하게 잘 나온다고 합니다. 

#### (2) Source code 

소스코드는 https://sites.google.com/a/deepmind.com/dqn/ 에 있으며 비상업적 용도로만 사용 가능합니다. 

#### (3) Model architecture

입력은 state만 주고, 이 state에서 가능한 모든 action에 대한 Q-value를 출력하게 함으로써 single forward pass만으로 Q-value를 연산하도록 하였습니다.

Model architecture는 2013년 버전에서는 2개의 convolutional layer를 사용한데 반해, 2015년 버전에는 3개를 사용하였으며, 각 conv layer의 필터수가 2배로 증가하였고 fully connected layer의 레이어의 수도 2배로 증가하였습니다. 다시 말해 model architecture가 2배 이상으로 커졌습니다. 

2013년 버전 아키텍처

* Input layer: 함수 Φ에 의해 생성된 84 x 84 x 4 의 이미지
* Conv layer #1: 16 filters of 8 x 8 with stride 4 + ReLU
* Conv layer #2: 32 filters of 4 x 4 with stride 2 + ReLU
* Fully connected layer: 256 + ReLU
* Output layer: Fully connected layer with a single output for each valid action (4개 ~ 18개)

2015년 버전 아키텍처

* Input layer: 함수 Φ에 의해 생성된 84 x 84 x 4의 이미지
* Conv layer #1: 32 filters of 8 x 8 with stride 4 + ReLU
* Conv layer #2: 64 filters of 4 x 4 with stride 2 + ReLU
* Conv layer #3: 64 filters of 3 x 3 with stride 1 + ReLU
* Fully Connected layer: 512 + ReLU
* Output layer: Fully connected layer with a single output for each valid action (4개 ~ 18개)

<figure>
	<img src="/img/3/architecture.png" alt="alt text">
</figure>	

#### (4) Training details 

(DQN의 성능 vs. 기존 알고리즘의 성능 vs. 사람의 성능) DQN 의 성능을 기존 알고리즘의 성능, 사람과의 성능과 비교한 결과가 아래 그림에 나와있습니다. 2013년에는 아타리 2600 게임 7개에 대해 성능을 테스트하였으나, 이번 논문에서는 총 49개의 게임에 대해 비교하였습니다. 

그림에서 각 게임에 나와있는 수치는 (DQN의 점수 - 랜덤플레이 점수)/ (프로게이머의 점수 - 랜덤플레이 점수) 에 100을 곱해 계산한 것입니다. 다시 말해 0%는 DQN의 점수 (파란색 부분) 가 랜덤플레이의 수준이라는 의미이며, 100%는 프로게이머 수준이라는 것입니다. 일반적인 게이머의 수준은 프로게이머의 75% 수준으로 가정하였습니다. 회색 부분은 기존의 선형 함수 기반 학습 알고리즘을 이용하여 테스트했을 때의 점수를 의미합니다. 

<figure>
	<img src="/img/3/atari_result.png" alt="alt text">
</figure>	

모든 게임에 대해 동일한 네트워크 아키텍처, 학습 알고리즘, 하이퍼파라미터를 사용하였습니다. 하이퍼파라미터의 설정은 아래 표를 참고하도록 합니다. 

<figure>
	<img src="/img/3/hyperparameters.png" alt="alt text">
</figure>	

표에서 보다시피 대부분의 하이퍼파라미터의 설정은 2013년 연구의 설정과 대동소이합니다. 특이한 파라미터로는 1) target Q-network의 업데이트는 10,000번마다 진행한다는 점, 2) replay는 50,000개의 replay가 메모리에 쌓인 다음부터 활용하기 시작한다는 점 (즉, 50,000개가 쌓이기까지는 랜덤 플레이), 3) 에피소드 시작 시 에이전트는 최대 30번까지 no-op (아무런 동작도 하지 않음) 하게 한다는 것 정도입니다. 

게임에 준 유일한 변형은 학습 시의 보상 체계입니다. 모든 양의 보상은 +1, 음의 보상은 -1로 설정함으로써 reward clipping을 적용한 것입니다. Reward clipping 관련된 자세한 내용은 직전에 포스팅한 2013년 연구에 대한 리뷰를 참고하기 바랍니다. 

신경망의 optimizer로는 RMSProp 알고리즘을 사용하였으며, 미니배치의 수는 32를 사용하였습니다. 학습중의 behavior policy로는 ε-greedy 방식을 사용하였는데, 역시 자세한 내용은 2013년 연구 리뷰를 참고하면 됩니다. 

학습은 2013년에는 1,000만 프레임에 대해 진행했는데, 이번에는 5,000만 프레임에 대해 진행하였습니다. 5,000만 프레임은 게임 시간으로 따지면 38일동안 꼬박 게임을 한 것과 동일한 분량입니다. Replay memory에는 가장 최근 프레임이 100만개 포함될 수 있도록 하였습니다. 

Frame skipping 기술도 활용을 하였는데, 역시 상세 내용은 2013년 연구 리뷰를 참고하시기 바랍니다.

하이퍼파라미터 값과 optimization 파라미터 값은 Pong, Breakout, Seaquest, Space Invaders, Beam Rider 등 5개의 게임에 대한 informal search를 통해 정했습니다. 정식으로 하이퍼파라미터 값을 정하기 위해서는 systematic grid search를 수행해야 했지만, 연산 비용이 너무 높기 때문에 수행하지 못 했습니다. 

실험은 아래와 같은 게임에 대한 최소한의 지식만을 이용하여 진행하였습니다.

* 화면 이미지
* 점수 (게임마다 다르나 변형 없이 사용)
* 액션의 수 
* 생명의 수 (예를 들어 Breakout의 경우에는 5개)

#### (5) Evaluation procedure

학습을 마친 에이전트는 각각의 게임을 30회씩 (최대 시간 5분) 플레이하여 테스트를 진행하였습니다. 테스트 시에 오버피팅이 발생하는 가능성을 최소화하기 위해 ε-greedy policy를 적용하였으며, 이때 ε의 값은 0.05를 사용하였습니다. 

Baseline comparison으로는 랜덤 에이전트를 사용하였는데, 사람이 어떤 버튼을 누를 수 있는 가장 빠른 속도인 10Hz (6번째 프레임) 마다 랜덤한 액션을 선택하도록 하고 이후 다음 선택 지점 (6 프레임 후) 까지는 동일한 액션을 반복하도록 하였습니다. 사실 60Hz (매 프레임) 마다 액션을 선택하도록 랜덤 에이전트를 실험하기도 하였으나 겨우 6개의 게임에서만 성능이 5% 정도 향상되었을 뿐 특별한 효과는 없었습니다. 

사람 테스터는 중간에 게임을 멈추거나 저장하거나 리로드할 수 없게 하였으며 게임 사운드는 끈 상태에서 진행함으로써 에이전트의 환경과 유사하게 맞추어 주었습니다. 사람의 성능은 각각의 게임에 대해 2시간동안 연습하게 한 다음에 실제 테스트는 20번 (최대 시간 5분) 진행하여 측정하였습니다. 

#### (6) Algorithm

알고리즘은 2013년 논문 리뷰의 2. Background 부분과 동일하니, 이를 참고하기 바랍니다. 

#### (7) Training algorithm for DQN

전체 알고리즘은 아래와 같습니다. 

<figure>
	<img src="/img/3/algorithm.png" alt="alt text">
</figure>	

2013년 연구 당시의 알고리즘과 비교하면 딱 한가지 차이만 존재합니다. 바로 target Q-network의 설정과 관련된 3행과 15행 입니다.  

* 3행에서는 target action-value function Q hat 의 가중치 θ-를 θ로 초기화하고 있으며, 
* 15행에서는 C step마다 Q 을 Q로 리셋해 줍니다.

기존의 Q-learning 알고리즘과의 차이점은 크게 아래 세가지입니다. 

#### (1) Experience replay의 사용

Experience replay 관련된 알고리즘에 대해서는 2013년 논문에서 했던 언급을 그대로 다시 언급하고 있기 때문에 2013년 연구 리뷰를 참고하기 바랍니다. 

#### (2) Target Q-network

Target Q-network 관련해서는 앞의 0. Novelty 부분의 설명을 참고하도록 합니다. 

#### (3) Error Clipping

Error (loss) 를 -1과 1 사이에 오도록 하는 것이 알고리즘의 안정성에 도움이 된다는 사실을 발견하였습니다. Error clipping은 Huber loss function을 이용하여 구현 가능합니다. 


### 2. 성능 비교 (Experience replay, Target Q-Network)

아래 표에서 보다시피 experience replay와 target Q-network를 가미함으로써 성능이 비약적으로 향상된 것을 확인할 수 있습니다.

<figure>
	<img src="/img/3/performance.png" alt="alt text">
</figure>	

### 3. 성능 비교 (DQN vs. Linear function approximator)

대부분의 게임에서 DQN은 linear function approximator보다 월등한 성능을 보였습니다. 

<figure>
	<img src="/img/3/performance2.png" alt="alt text">
</figure>	

### 4. Value function 시각화

#### (1) Breakout에서의 value function 추이

게임 화면의 블록을 자세히 보게되면, 아래쪽 블록을 깨는 ①과 ②에서는 17 정도의 state value를 보이며, 위쪽 블록을 깨는 ③에서는 state value가 21 정도로 올라가는데, 그 이유는 이제 터널을 뚫고 여러개의 블록을 한꺼번에 깰 수 있다는 기대감이 반영되었기 때문입니다. 마지막으로, 터널을 뚫고 들어간 ④에서는 state value가 23 이상으로 올라가는 것을 확인할 수 있습니다. 

<figure>
	<img src="/img/3/value_fn_breakout.png" alt="alt text">
</figure>	

#### (2) Pong 에서의 action-value function (Q-value function) 추이

아래 그림을 보면, ①에서는 NO-OP, UP, DOWN에 대한 Q-value가 모두 0.7 수준입니다. ②에 가면 위로 이동해서 공을 맞춰야 하기 때문에 UP의 Q-value가 높고, NO-OP과 DOWN의 Q-value는 매우 낮은 것을 확인할 수 있습니다. ③에서는 조금 더 위로 올라가야 하기 때문에 여전히 UP의 Q-value가 높게 나타납니다. ④에서는 공이 상대방의 paddle을 넘어서 보상 +1을 따기 직전이기 때문에 NO-OP, UP, DOWN 모두에 대해 Q-value가 1로 나타납니다. 

<figure>
	<img src="/img/3/value_fn_pong.png" alt="alt text">
</figure>	

### 5. t-SNE 시각화 1

아래 그림은 t-SNE 알고리즘을 실행하여 DQN의 마지막 hidden layer를 표현한 것입니다. 각 점의 색은 각 state에 대한 state value (V, state의 최대 기대 보상) 를 나타내며 진홍색은 높은 state value를, 군청색은 낮은 state value를 의미합니다. 

게임 화면이 가득 차 있는 경우 (우측 상단) 와 거의 비어 있는 경우 (좌측 하단) 에는 state value가 높은데, 그 이유는 우리의 DQN이 화면을 비우면 (즉, 게임을 클리어하면) 화면이 가득 찬 화면으로 이어진다 (즉, 새로운 다음 게임이 시작된다) 는 점을 학습하였기 때문입니다. 이에 비해 중간쯤 진행된 경우 (중앙 하단) 에는 즉각적인 보상이 작기 때문에 state value가 낮습니다. 

우측 하단, 좌측 상단, 중앙 상단의 게임화면을 보면 앞의 예들과는 다르게 화면의 구성에 다소 차이가 있음에도 불구하고 유사한 state value를 가집니다. 그 이유는 게임 화면을 자세히 보면 오렌지색의 벙커의 존재 여부에 의해 달라보이는 것인데, 이 벙커는 게임을 클리어하는데 그다지 중요하지 않기 때문입니다. 

<figure>
	<img src="/img/3/t-sne1.png" alt="alt text">
</figure>	

### 6. t-SNE 시각화 2

이 그림은 사람의 플레이와 DQN의 플레이의 조합에 대해 t-SNE 알고리즘을 실행하여 DQN의 마지막 hidden layer를 표현한 것입니다. 사람의 플레이는 오렌지색, DQN의 플레이는 파란색으로 표현하였는데, 유사한 state 를 가지는 화면은 사람의 플레이와 DQN의 플레이에서 유사하게 나타난다는 점에서 DQN에 의해 학습된 표현이 실제로 일반화하고 있다는 것을 보여주고 있습니다. DQN 에이전트가 사람의 플레이에서 발견되는 것과 유사한 state의 시퀀스를 따른다는 것입니다. 

<figure>
	<img src="/img/3/t-sne2.png" alt="alt text">
</figure>	
