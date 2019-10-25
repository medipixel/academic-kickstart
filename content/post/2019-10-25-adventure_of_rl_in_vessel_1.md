+++
title = "강화학습의 혈관 속 탐험 - 로봇 제어와 강화학습"
summary = "Introduction of guide-wire control for PCI by RL - Why we use RL?"
date = 2019-10-25T10:00:00+09:00
draft = false
authors=["kyunghwan-kim"]
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

이 글에서는 왜 저희가 관상동맥 중재술 로봇 제어 방법으로 강화학습을 선택하게 되었는지 얘기해보고자 합니다. 이미 수많은 학자와 엔지니어들이 로봇을 위한 효율적이고 뛰어난 제어 이론들을 개발해왔습니다. 이러한 제어 이론들이 어떠한 점에서 강화학습과 다르고, 어떠한 장단점으로 인해 저희가 강화학습을 선택하게 되었는지 이야기해보겠습니다. 제일 먼저, 제어 이론 중 가장 보편적이고 널리 쓰이는 PID (Proportional-Integral-Derivative) 제어와의 비교로 시작하겠습니다.