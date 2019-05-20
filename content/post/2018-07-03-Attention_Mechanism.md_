---
header: Attention Mechanism 
mathjax: True
---
딥러닝 분야에서 많이 연구가 이뤄지고 있는 Attention Mechanism을 소개합니다. 2개의 포스트로 내용을 나누어 첫번째 포스트에서는 기본적으로 알려진 Attention Mechanism의 개념, 직관 및 예시를 다루고 두번째 포스트에서는 Self-Attention의 개념, 직관 및 예시를 다룹니다.


<!--break-->
***

### What is Attention?
아래 글은 Wikipedia에 기술되어 있는 Attention의 의미[[1]](https://en.wikipedia.org/wiki/Attention)입니다. 아래에 깜빡이는 2개의 그림을 각각 보고 Attention의 의미를 설명한다고 했을 때, 어떤게 더 쉬우신가요? 
<figure>
	<img src="/img/3/attn.gif" alt="alt text">
	<figcaption>그림0. Attention YES/NO</figcaption>
</figure>	


놀랍게도 중요한 정보(단어)에 표시를 했을 때 그렇지 않은 경우보다 정보를 받아들이는 속도나 정도가 좋아집니다. 이 것이 Attention이 가지는 힘입니다. 

딥러닝에서 사용되는 Attention은 어떨까요? 마찬가지입니다. 입력되는 많은 정보들 중에 우리가 알고자하는 내용에 좀 더 가중치를 주자는 것으로 보시면 됩니다. 

### Why Attention?
딥러닝에서는 RNN, CNN 등 많은 모델 구조들이 사용되고 있습니다. 이러한 구조들은 다양한 문제들을 잘 풀지만 몇 가지 문제점들이 있습니다. 어떤 문제점들이 있고 Attention이 이런 문제점들을 어떤 방식으로 해결 할 수 있는지 살펴보겠습니다. 예시는 위의 문단을 활용하겠습니다.

#### 1) Recurrent Neural Network(RNN)
RNN은 단어를 순서대로 보면서 전체 문단의 정보를 순차적으로 살핍니다. 마지막까지 읽었을 때 모든 정보가 종합이 되고 여기에서 의미를 이끌어냅니다. 
사람이 RNN의 방식으로 읽었다고 생각하면 어떨까요? 짧은 글은 쉽게 의미를 파악 할 수 있을 것지만 길이가 길어지면 정보를 잊어버릴 것 같습니다. 특히 앞의 정보를요. 

<figure>
	<img src="/img/3/rnn.gif" alt="alt text">
	<figcaption>그림1. RNN이 정보를 처리하는 방식</figcaption>
</figure>	


이런 문제가 딥러닝에서도 마찬가지로 일어납니다. 번역 문제에서 RNN의 성능이 문장 길이가 일정 수준 이상 되는 경우 급격하게 성능 하락이 생기는 것을 실험적으로 확인 할 수 있습니다.[[2]](https://arxiv.org/abs/1409.0473.pdf) 

이 외에도 한 가지 문제가 더 있습니다. RNN처럼 정보를 처리하는 경우 문장 내 단어 순서대로 정보를 처리해야 하기 때문에 시간에 대한 의존성이 생기게 됩니다. 그럴 경우 전체 문단의 정보를 병렬적으로 한꺼번에 연산 할 수 없기 때문에 비효율이 발생합니다.[[3]](https://arxiv.org/pdf/1706.03762.pdf)

#### 2) Convolutional Neural Network(CNN)
CNN은 로컬한 영역을 필터로 살피면서 정보를 추상화해갑니다. 다음 layer에서는 이전에 추상화된 정보를 받아 더 추상화하는 식으로 진행이 됩니다. 이러한 작동 방식 때문인지 이미지 처리에 큰 강점을 보입니다. 
<figure>
	<img src="/img/3/cnn.gif" alt="alt text">
	<figcaption>그림2. CNN이 정보를 처리하는 방식</figcaption>
</figure>	

그럼 위 예시에서와 같이 개별적인 단어들 사이의 관계를 학습해서 의미를 이끌어 내야 하는 경우는 어떨까요? 사람이 CNN의 방식으로 읽었다고 생각해봅시다. 특정 영역을 보기 때문에 문장 사이의 행간 의미를 파악하는데 도움이 조금은 될 듯 하지만 보통은 문장 단위나 단어의 순서를 통해 의미를 파악하는 경우가 많아 혼란이 올 것 같기도 합니다. 

이런 방식의 장점도 있습니다. RNN과 같이 시간 축 순서대로 연산을 하지 않아도 되기 때문에 모든 입력 값을 병렬적으로 연산 할 수 있습니다. 그래서 ByteNet[[4]](https://arxiv.org/abs/1610.10099.pdf), ConvS2S[[5]](https://arxiv.org/abs/1705.03122.pdf)와 같은 CNN 기반의 모델이 제시가 되긴 하지만 로컬 영역을 쌓으면서 receptive field를 넓히고 이를 통해 단어들 사이의 관계를 학습하기 때문에 layer가 많아질 수 밖에 없고 그로 인한 많은 연산을 및 학습 문제가 발생하게 됩니다. [[6]](http://www.bioinf.jku.at/publications/older/ch7.pdf)

#### 3) Attention Mechanism 
Attention을 도입하면 어떻게 될까요? 자세한 메커니즘은 뒤에서 보도록 하고 결론부터 살펴보시죠.
그림처럼 전체 문단에 대해서 설명해야 하는 Attention의 의미와 관련된 정보만 가중치(학습이 됩니다.)를 준 형태로 받아들입니다. 
<figure>
	<img src="/img/3/attention.png" alt="alt text">
	<figcaption>그림3. Attention이 정보를 처리하는 방식</figcaption>
</figure>	

Attention을 사용하면 RNN와 같이 시간에 대한 의존성이 없기 때문에 병렬 연산이 가능하며 단어들 간의 의존성을 학습하는 데에 CNN와 같이 너무 많은 layer를 쌓지 않아도 됩니다. RNN, CNN와 비교했을 때 단어들 간의 관계를 직접 학습하기 때문에 문단 내 거리가 먼 단어들 사이의 의존성도 학습이 잘 됩니다. 

***
### Attention Mechanism
수식, 그림으로 기본적인 Attention Mechanism이 어떻게 이루어지는지 살펴보겠습니다.
아래 수식은 많은 논문들에서 공통적으로 정의하는 Attention을 나타냅니다.

$$\text{Attention}(Q,K,V) = \text{softmax}(\dfrac {QK^T} {\sqrt d_k})V$$ 
$$(Q: \text{Query},\ K: \text{Key},\ V: \text{Value},\ d_k: \text{dimension of K})$$
<figure>
	<img src="/img/3/attention_mechanism.png" alt="alt text">
	<figcaption>그림4. Attention Mechanism</figcaption>
</figure>	

$Q, K, V$는 어떤 특정 정보이며 $K, V$는 쌍으로 같이 다니는 값이라고 생각하시면 됩니다.
Attention을 통해서 저희가 최종적으로 하고 싶은 건 $V$ 정보에 가중치를 줘서 필요한 정보를 많이 얻는 것 입니다.(앞서 문단에 진하게 표시된 단어들!). 어떻게 얻는지 아래 수식을 통해 살펴 보겠습니다.

- $QK^T$: $V$의 쌍으로 있는 $K, Q$라는 어떤 정보와의 관계(혹은 유사도)를 구합니다.
- $\dfrac {QK^T} {\sqrt{d_k}}$: 특정 값만 너무 크면 가중치를 구했을 때 해당 값이 너무 지배적이게 됩니다. 그래서 normalize를 해줍니다.
- $\text{softmax}(\dfrac {QK^T} {d_k})$: 유사도를 가중치로 바꿔줍니다.
- $\text{softmax}(\dfrac {QK^T} {d_k})V$: 구한 유사도를 $V$에 반영해줍니다.

즉, $V$에서 필요한 부분에 가중치를 주고 싶은데, 이를 $V$의 쌍인 $K$와 $Q$와의 관계를 통해서 정의하고 싶은 것입니다. 그림은 수식을 표현해놓았고 여기에서 mask는 무시하셔도 됩니다.


많은 논문들에서 $K,V$는 같게 두는 경우가 많으며 여기서 가장 중요한 점은 $Q, K$의 관계를 어떻게 정의 할 것이냐 입니다. 관계가 유의미해야 가중치가 유의미해 질 것이니까요. 어떤 식으로 논문들에서 고려하는 지는 예제로 살펴보도록 하겠습니다.

#### 1) 예시1. Seq2Seq with Attention
기계 번역에 많이 사용되는 구조인 Seq2Seq [[7]](https://arxiv.org/pdf/1409.3215.pdf)을 간략하게 살펴보고 어느 부분에 Attention을 사용 할 수 있는지 [2] 를 보도록 하겠습니다. 

Seq2Seq2는 기본적으로 Encoder, Decoder로 구성됩니다. Encoder에서 영어를 입력 받은 뒤 정보를 Decoder에 넘겨줘서 한글로 번역하는 과정을 거칩니다. 앞에서, RNN이 문장이 길어질 때 겪는 어려움을 봤습니다. Encoder, Decoder에도 RNN이 사용되기 때문에 문장이 길어졌을 때 발생하는 문제에 Attention이 도움이 될 수 있을 지 살펴보겠습니다.

RNN에 대해서 잠깐 살펴보시죠. Encoder, Decoder는 아래 그림 및 수식으로 구성이 됩니다. 
보시면 Decoder와 Encoder의 hidden state($s_{t'}, h_{t}$)는 모두 이전 시간의 hidden state($s_{t'-1}, h_{t-1}$)에 의존적입니다. 시간 축을 따라서 Encoder의 정보가 쭉 전달되고 이를 Decoder에서 받아서 쭉 전달하는 형태입니다.

<figure>
	<img src="/img/3/seq2seq.png" alt="alt text">
	<figcaption>그림5. Basic Seq2Seq 구조</figcaption>
</figure>	

앞서 RNN의 일정 길이 이상의 정보 전달 시에 성능이 하락한 것을 살펴봤습니다. 이로부터  Encoder의 정보가 Decoder에 전달될 때 손실이 생긴다고 가정을 해보죠. 그럼, Encoder의 정보를 직접적으로 전달해주는 방법을 생각해 볼 수 있지 않을까요?  Attention으로요. 

근데 위에서 살펴 본 것처럼 Attention은 $Q, K, V$가 필요합니다. 정해보겠습니다.

전달해 줄 정보는 Encoder의 정보이므로 $V=K=(h_1, h_2, ..., h_t)$가 될 것이고 $V$는 $K$와 같다고 놓습니다. $Q$는 Output을 결정하는 직접적인 값이 되어야 가중치를 통해 부족한 정보를 더 전달해 줄 수 있을 것입니다. 따라서 $Q$는 Decoder의 hidden state인 $s_{t'}$로 정합니다. $s_{t'}$이 $s_{t'+1}$를 구할 때 사용되기 때문에, Encoder의 정보와의 관계를 고려해서 $s_{t'+1}$에 필요한 정보를 더 전달해 줄 수 있게 됩니다. 


<figure>
	<img src="/img/3/seq2seq_attention.png" alt="alt text">
	<figcaption>그림6. Seq2Seq with Attention</figcaption>
</figure>	

결과적으로 위와 같은 형태의 모델이 되며 성능을 비교했을 때 RNN에 비해 입력 문장이 길어질 경우 성능 하락이 크지 않습니다. (RNNsearch이 attention이 적용된 모델입니다.)

<figure>
	<img src="/img/3/attention_perform.png" alt="alt text">
	<figcaption>그림7. Attention 유무에 따른 성능 비교</figcaption>
</figure>	


#### 2) 예시2. Style-token
약간 문제를 바꿔서 TTS(Text To Speech) 모델에서 Attention이 어떻게 사용되었는지 살펴보도록 하겠습니다. 
Style-Token을 활용하면  텍스트를 음성으로 변환이 가능하며 이 때, 사람의 음성 스타일(억양, 톤,...)을 입혀줄 수 있습니다.([데모](https://google.github.io/tacotron/publications/global_style_tokens/)) 

모델을 간략하게 소개하자면 학습 시에 입력 값으로 텍스트와 오디오가 함께 들어옵니다. 오디오는 단순 Loss를 구하기 위한 label 역할을 한다고 보셔도 되고 텍스트가 위에서 소개한 Seq2Seq with Attention 구조를 통과하면서 오디오를 생성합니다. 이는 기본적인 Tacotron 구조[[8]](https://arxiv.org/pdf/1703.10135.pdf)로 TTS 성능이 좋긴 하지만 특정 사람의 스타일은 학습이 되지 않아 기계적인 음성이 주로 생성됩니다. 

여기에 Attention을 사용해서 문제를 접근했고 데모에서 보신 것처럼 사람들의 특성을 스타일로 입히는데 성공합니다.

그럼, 어디에 Attention이 사용된 것인지 살펴보도록 하겠습니다.

아래 그림은 논문에 나와있는 Style-token 구조입니다. 여기에서 우리가 하고자 하는 목표는 음성에서 어떤 style에 해당되는 embedding feature를 분리해내는 일입니다. 
이 embedding feature는 어떤 token(latent vector)들의 조합으로 이루어진다고 가정을 하면 우리는 token을 random하게 생성한 뒤 조합하는 방식을 통해 스타일을 나타낼 수 있습니다. 이를 GST(Global Style Token)이라고 합시다. 
우린 GST를 가지고 어떤 전달하고자 하는 정보를 만들 것이기 때문에 $V=K=$ GST가 될 것입니다. 이 때도, $V$와 $K$는 같은 값으로 두겠습니다.

자, 그럼 이제 $K$와 연관이 있는 $Q$만 정해주면 됩니다. 각 사람들 음성의 스타일은 그 사람의 목소리에서 나오는 것이기 때문에 이와 연관이 있다고 보면 괜찮지 않을까요? 그래서 오디오 정보($Q$)와 GST($V=K$) 사이의 관계를 통해 가중치를 구하고 이를 GST에 반영해주면 전달해주고자 하는 Attention 값이 됩니다.

Attention 값은 오디오 특정 화자의 스타일의 정보를 가지고 있기 때문에 Encoder에 넣어주게 되면 스타일이 입혀진 목소리를 생성할 수 있게 됩니다.

<figure>
	<img src="/img/3/style_token.png" alt="alt text">
	<figcaption>그림8. Style-token 구조</figcaption>
</figure>	


### Conclusion
Attention이 딥러닝에서 사용될 때의 방식과 직관을 Seq2Seq, Style-token 2가지 예시를 통해 살펴봤습니다. 저는 Attention이라는 개념을 정보들 사이에 어떤 관계를 정의해 줄 것인가를 통해 특정한 정보를 더 잘 전달하는 방법이라고 생각합니다. 글이 길어져서 다음 글에서 Self-attention은 무엇인지 살펴보도록 하겠습니다.

### Reference
[1] [Attention - Wikipedia](https://en.wikipedia.org/wiki/Attention) <br>
[2] [Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio. Neural Machine Translation by Jointly Learning to Align and Translate. ArXiv. 2014](https://arxiv.org/abs/1409.0473.pdf)<br>
[3] [A. Vaswani, N. Shazeer, N. Parmar, and J. Uszkoreit. Attention is all you need. ArXiv, 2017.](https://arxiv.org/pdf/1706.03762.pdf)<br>
[4] [Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Koray Kavukcuoglu. Neural machine translation in linear time. ArXiv. <br>
2017](https://arxiv.org/abs/1610.10099.pdf)<br>
[5] [Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. ArXiv. 2017.](https://arxiv.org/abs/1705.03122.pdf)<br>
[6] [Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.](http://www.bioinf.jku.at/publications/older/ch7.pdf)<br>
[7] [Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural networks. ArXiv. 2014.](https://arxiv.org/pdf/1409.3215.pdf)<br>
[8] [Y. Wang, R. Skerry-Ryan, D. Stanton, Y. Wu, R. J. Weiss, N. Jaitly, Z. Yang, Y. Xiao, Z. Chen, S. Bengio, et al.  Tacotron: Towards end-to-end speech synthesis. ArXiv. 2017.](https://arxiv.org/pdf/1703.10135.pdf)<br>
[9] [Style-Token demo](https://google.github.io/tacotron/publications/global_style_tokens/)<br>
