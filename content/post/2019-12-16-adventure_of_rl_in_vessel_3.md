+++
title = "강화학습의 혈관 속 탐험 (3) - 실험환경 구성과 강화학습 알고리즘 소개"
summary = "Introduction to guide-wire control for PCI by RL (3) - How to apply RL"
date = 2019-11-27T10:00:00+09:00
draft = false
authors=["kyunghwan-kim", "chaehyeuk-lee"]
# Tags and categories
# For example, use `tags = []` for no tags, or the form `tags = ["A Tag", "Another Tag"]` for one or more tags.
tags = []
categories = []

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
[image]
  # Caption (optional)
  caption = ""

  # Focal point (optional)
  # Options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
  focal_point = ""
  
+++

#### 강화학습의 혈관 속 탐험 시리즈 \
[강화학습의 혈관 속 탐험 (1) - 로봇과 심혈관 중재 시술]({{< ref "2019-10-25-adventure_of_rl_in_vessel_1.md" >}})\
[강화학습의 혈관 속 탐험 (2) - 강화학습과 제어 이론의 비교]({{< ref "2019-11-26-adventure_of_rl_in_vessel_2.md" >}})

## Intro
---
강화학습으로 특정 task를 학습하기 위해서는 에이전트가 될 **강화학습 알고리즘을 선정**하고 에이전트가 상호작용할 **환경을 구성**해야합니다. 이번 포스트에서는 저희가 강화학습으로 PCI 로봇을 제어하기 위해 사용한 알고리즘과 환경에 대해 소개하겠습니다.

- [실험 환경 구성] (#ref_11)
  - [PCI 시술 환경] (#ref_12)

<a id="ref_11"></a>

## 실험 환경 구성
---
강화학습 **에이전트(Agent)**는 아래 그림과 같이 **상태(state)**에서의 **행동(action)**을 선택하고 얻는 **보상(reward)**을 통해 학습합니다. 이때 상태와 보상 등의 정보들을 **환경과의 상호작용**을 통해 얻습니다. 이러한 학습 프로세스를 그림으로 표현하면 아래 그림과 같습니다. 환경이 제대로 구성되지 않으면 어떤 강화학습 알고리즘을 쓰더라도 제대로 학습할 수 없습니다. 그렇기 때문에 강화학습에서 환경은 아주 중요한 개념입니다. 강화학습의 환경은 주로 우리가 풀고 싶은 문제를 **시뮬레이터**로 제작하여 가상 환경으로 구성하는 방법과 해당 문제를 실제 **real world**에서 반복 가능한 형태로 실험 환경을 구성하는 방법이 있습니다. 저희는 PCI에서의 가이드와이어 제어 문제를 풀고자 하기 때문에 시술 과정을 강화학습 에이전트가 상호작용 가능한 환경의 형태로 만들 필요가 있습니다. 이를 위해 PCI의 과정을 알아봅시다.

<center><img src="https://user-images.githubusercontent.com/17582508/69700749-938ed280-112e-11ea-81a1-281ce684b860.png" width="80%"></center>

<a id="ref_12"></a>

### PCI 시술 환경
---
PCI 시술에서 시술자가 환자의 혈관 모양과 현재 가이드와이어의 위치를 X-ray 조영 영상을 통해 봅니다. 현재 병변 위치가 어디인지, 제어하고 있는 가이드와이어의 모양과 위치가 어디인지 등의 정보를 모두 2D 이미지를 통해 확인합니다. 시술자는 이미지 정보를 바탕으로 가이드와이어를 어떻게 움직여야 할지 판단합니다. 가이드와이어를 움직이면 다시 조영 영상이 바뀌게 되고 시술자는 영상을 보고 다시 와이어를 움직입니다. 숙련된 시술자는 이러한 과정을 굉장히 빠르게 수행하여 환자의 병변 부위를 치료합니다.
이런 프로세스를 위의 강화학습 프로세스에 맞게 표현하면 다음 그림과 같습니다 [[1]](#ref_1). 시술자는 에이전트, 환자의 혈관과 X-ray 시술 장비가 환경, X-ray 조영 영상이 상태, 가이드와이어를 움직이는 것이 행동이 됩니다.
<center><img src="https://user-images.githubusercontent.com/17582508/81290036-50a03600-90a2-11ea-842f-2941dd8de973.png" width="80%"></center>


### 혈관 모형을 이용한 실험 환경 구성
---
이전 포스트에서 언급한 것처럼 PCI 시술은 가이드와이어의 마찰, 사람의 호흡, 심장의 박동 등의 이유로 모델링이 굉장히 어렵습니다. 모델링이 어렵다는 것은 다시 말해 시뮬레이터와 같은 가상 환경으로 제작하기 어렵다는 뜻입니다. 따라서 저희는 시뮬레이터보다는 **혈관 모형을 이용한 real world 실험 환경**을 구성하였습니다. 실제 시술에서 의사가 X-ray 이미지를 보고 가이드와이어를 어떻게 제어할 지 판단하듯이 강화학습 에이전트는 2D 혈관의 이미지를 보고 가이드와이어의 제어를 학습하는 것입니다. 이를 위의 시술 과정 프로세스와 같이 표현하면 아래 그림과 같습니다.
<center><img src="https://user-images.githubusercontent.com/17582508/81290394-f2278780-90a2-11ea-9cff-1135576c58f7.png" width="80%"></center>

이렇게 구성한 2D 혈관 모형 환경에서는 

## 알고리즘 선정
---

### Discrete action vs Continuous action
---

### Human demo 활용
---

## 학습 결과
---

## 마치며
---


## Reference
---
<a id="ref_1"></a>
**[1]** CAG 이미지 출처: https://www.researchgate.net/figure/CAG-images-of-the-first-PCI-a-Coronary-stenosis-in-the-proximal-mid-portion-of-LAD_fig1_316498381 \
