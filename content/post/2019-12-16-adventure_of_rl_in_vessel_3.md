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
강화학습 에이전트는 아래 그림과 같이 **상태(state)**에서의 **행동(action)**을 선택하고 얻는 **보상(reward)**를 통해 학습합니다. 이때 상태와 보상 등의 정보들을 **환경과의 상호작용**을 통해 얻습니다. 환경이 제대로 구성되지 않으면 어떤 강화학습 알고리즘을 쓰더라도 제대로 학습할 수 없습니다. 그렇기 때문에 강화학습에서 환경은 아주 중요한 개념입니다. 강화학습의 환경은 주로 우리가 풀고 싶은 문제를 **시뮬레이터**로 제작하여 가상 환경으로 구성하는 방법과 해당 문제를 실제 **real world**에서 반복 가능한 형태로 실험 환경을 구성하는 방법이 있습니다. 

<center><img src="https://user-images.githubusercontent.com/17582508/69700749-938ed280-112e-11ea-81a1-281ce684b860.png" width="80%"></center>

<a id="ref_12"></a>

### PCI 시술 환경

### 혈관 모형을 이용한 실험 환경

## 알고리즘 선정

### Discrete action vs Continuous action

### Human demo 활용

## 학습 결과

## 마치며
---


## Reference
---
<a id="ref_1"></a>
**[1]** Wikipedia (PID Controller) https://en.wikipedia.org/wiki/PID_controller \
