+++
title = "강화학습의 혈관 속 탐험 (1) - 로봇과 심혈관 중재 시술"
summary = "Introduction to guide-wire control for PCI by RL - PCI aided by Robot"
date = 2019-10-25T10:00:00+09:00
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

이번 포스트에서는 저희가 연구 중인 **강화학습을 이용한 심혈관 중재술 로봇 제어**에 대해 소개하려고 합니다. 저희가 연구 중인 주제를 들으면 "심혈관 중재술이 뭐지?", "왜 로봇을 제어를 하는데 강화학습을 쓰지?" 와 같이 여러 궁금증이 먼저 생기실 것입니다. 그래서 포스트를 통해 이런 궁금증들을 해소하는 시간을 가지려 합니다.

## 심혈관 중재술?
---
저희는 심혈관 질환의 진단 및 치료를 돕는 솔루션을 연구 개발 하고 있습니다. 심혈관 질환에는 여러 종류가 있고 그에 따른 치료방법도 다양합니다. 그 중 현재 저희가 연구하는 내용과 가장 연관이 있는  **허혈성 심장 질환**과 이를 치료하기 위한 **심혈관 중재술**에 대해 설명하려 합니다.

### 허혈성 심장 질환
심장은 우리 몸 전체에 혈액을 보내는 펌프 역할을 하고 있습니다. 혈액을 통해 산소와 영양소를 공급하여 우리 몸이 에너지를 내고 생활을 할 수 있도록 합니다. 심장이 펌프 역할을 수행하려면 심장을 이루고 있는 심근에도 산소와 영양소를 공급해주어야 합니다. 이를 위해 심근에 혈액을 공급하는 혈관이 **관상동맥**입니다. 허혈성 심장 질환은 심장에 혈액을 공급해주는 혈관인 관상동맥에 지방질이 쌓여 혈관이 좁아지거나 막혀 심장근육의 일부에 혈액 공급이 부족하여 발생하는 질환입니다[[1]](#ref_1). 사람들에게 사망 원인 1위가 무엇인지 물으면 보통 암을 떠올리기 쉽지만 통계상으로 허혈성 심장 질환이 전세계 사망 원인 1위를 차지하고 있습니다. 그만큼 많은 사람들이 이 질환을 겪고 있으며 위험한 질환이라는 것을 알 수 있습니다. 허혈성 심장 질환에는 혐심증, 심근경색 등이 있습니다. 아래 이미지는 심혈관 질환의 예시를 보여줍니다[[2]](#ref_2).

![심혈관질환의 예시](https://user-images.githubusercontent.com/17582508/67544094-1c30e200-f72f-11e9-8aff-979103817187.png)

### 관상동맥 중재 시술(PCI)
허혈성 심혈관 질환을 치료하기 위한 방법 중 관상동맥 중재 시술(PCI)이 많은 비중을 차지하고 있습니다. PCI는 팔이나 다리의 혈관을 통하여 심장까지 들어간 다음, 막히거나 좁아진 심장 혈관을 뚫는 시술을 말합니다[[3]](#ref_3). 가슴을 절개하는 수술이 아니기 때문에 전신 마취가 필요없고 가슴에 흉터를 남기지 않아 비교적 환자의 부담이 덜한 치료방법입니다. 스텐트라는 시술 도구로 병변을 치료하기 때문에 일반적으로 스텐트 시술이라는 이름으로 많이 알려져 있습니다. 시술 방법은 다음과 같습니다[[4]](#ref_4).

1. 허벅지 위쪽 또는 팔의 혈관을 통해 **와이어**를 넣어 관상동맥의 병변부위에 접근
2. **벌룬**으로 막힌 혈관 부분을 넓히고, 병변에 따라 **스텐트**라는 그물망으로 고정

![image](https://user-images.githubusercontent.com/17582508/67552494-3de99380-f746-11e9-8404-6491e42f1339.png)

이 때, 시술자는 조영제와 X-ray를 이용한 **혈관조영영상**을 이용해 병변을 확인하고 시술 위치를 파악합니다. 시술이 진행되는 도중에도 계속해서 혈관조영영상을 확인하여 정확한 시술을 진행합니다. 아래는 혈관조영사진의 예시입니다[[5]](#ref_5).

![image](https://user-images.githubusercontent.com/17582508/67552998-324a9c80-f747-11e9-8b0b-b5c979a6a9a7.png)

## 심혈관 중재술 자동화의 필요성
---
PCI는 허혈성 심장 질환을 효과적으로 치료할 수 있는 시술 방법이지만 개선해야할 부분도 있습니다. 

1. 의료진의 방사능 피폭
    - ㅇ 

2. 긴 시술 시간

3. 숙련도에 따른 차이

이런 점들을 개선하기 위해 PCI 로봇에 대한 연구가 진행되고 있습니다. 대표적으로 Corindus의 원격 시술 로봇이 있습니다[[6]](#ref_6). 저희는 더나아가 로봇을 이용해 PCI를 자동화하는 연구를 진행하고 있습니다.


## Reference
---
<a id="ref_1"></a>
**[1]** 허혈성 심질환 설명(서울아산병원): http://www.amc.seoul.kr/asan/healthinfo/disease/diseaseDetail.do?contentId=30275

<a id="ref_2"></a>
**[2]** 심혈관질환의 종류 이미지(아스피린프로텍트): http://www.aspirinprotect.co.kr/ko/disease-and-prevention/cardiovascular-diseases/#tab_tab0

<a id="ref_3"></a>
**[3]** 관상동맥 중재 시술의 이해(서울아산병원 심장병원): http://heart.amc.seoul.kr/asan/depts/heart/K/bbsDetail.do?menuId=4634&contentId=264501

<a id="ref_4"></a>
**[4]** 이미지 출처: http://www.secondscount.org/image.axd?id=c8a00122-bb66-46c6-8ab7-333a9a0cd46a&t=635566481777430000, https://www.mcvs.co.nz/wp-content/uploads/2017/05/stent-balloon-angioplasty.png

<a id="ref_5"></a>
**[5]** 관상동맥 조영술 이미지 (강남세브란스병원): http://gs.iseverance.com/dept_clinic/department/cardiology/treatment/view.asp?con_no=82261

<a id="ref_6"></a>
**[6]** Corindus: https://www.corindus.com/