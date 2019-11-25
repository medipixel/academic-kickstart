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

이번 포스트에서는 저희가 연구 중인 **강화학습을 이용한 심혈관 중재시술 로봇 제어**에 대해 소개하려고 합니다. 처음 이 주제에 대해 들으시면 "심혈관 중재시술이 뭐지?", "왜 로봇을 제어를 하는데 강화학습을 쓰지?" 와 같이 여러 궁금증이 먼저 생기실 것입니다. 그래서 포스트를 통해 이런 궁금증을 해소하는 시간을 가지려 합니다.

## 심혈관 중재시술?
---
저희는 심혈관 질환의 진단 및 치료를 돕는 솔루션을 연구 개발 하고 있습니다. 심혈관 질환에는 여러 종류가 있고 그에 따른 치료방법도 다양합니다. 그 중 현재 저희가 연구하는 내용과 가장 연관이 있는  **허혈성 심장 질환**과 이를 치료하기 위한 **심혈관 중재시술**에 대해 설명하려 합니다.

### 허혈성 심장 질환
심장은 우리 몸 전체에 혈액을 보내는 펌프 역할을 하고 있습니다. 혈액을 통해 산소와 영양소를 공급하여 우리 몸이 에너지를 만들고 생활을 할 수 있도록 합니다. 심장이 펌프 역할을 수행하려면 심장을 이루고 있는 심근에도 산소와 영양소를 공급해주어야 합니다. 이를 위해 심근에 혈액을 공급하는 혈관이 **관상동맥**입니다 [[1]](#ref_1). 

<center><img src="https://user-images.githubusercontent.com/17582508/67763403-f71fd480-fa8a-11e9-9c49-a2ad9cd08634.png" width="40%"></center>

허혈성 심장 질환은 심장에 혈액을 공급해주는 혈관인 관상동맥에 지방질, 콜레스테롤, 칼슘 등이 쌓여 혈관이 좁아지거나 막혀 심장근육의 일부에 혈액 공급이 부족하여 발생하는 질환입니다 [[2]](#ref_2). 허혈성 심장 질환은 국내 사망 원인 통계를 보면 암에 이어 2위를 차지하고 있으며 전세계 통계에서는 1위를 차지하고 있습니다 [[3]](#ref_3). 그만큼 많은 사람들이 이 질환을 겪고 있으며 사망에 이를 정도로 위험한 질환이라는 것을 알 수 있습니다. 허혈성 심장 질환에는 협심증, 심근경색 등이 있습니다. 아래 이미지는 심혈관 질환의 예시입니다 [[4]](#ref_4).

<center><img src="https://user-images.githubusercontent.com/17582508/67544094-1c30e200-f72f-11e9-8aff-979103817187.png" width="90%"></center>

### 관상동맥 중재시술(PCI)
관상동맥 중재시술(PCI; Percutaneous Coronary Intervention)은 허혈성 심혈관 질환에서 큰 비중을 차지하고 있습니다. PCI는 팔이나 다리의 혈관을 통하여 심장까지 들어간 다음, 막히거나 좁아진 심장 혈관을 넓히는 시술입니다 [[5]](#ref_5). 가슴을 절개하는 수술이 아니기 때문에 전신 마취가 필요없고 가슴에 흉터를 남기지 않아 비교적 환자의 부담이 덜한 치료방법입니다. 스텐트라는 시술 도구로 병변을 치료하기 때문에 일반적으로 스텐트 시술이라는 이름으로 많이 알려져 있습니다. 시술 방법은 다음과 같습니다 [[6]](#ref_6).

1. 허벅지 위쪽 또는 팔의 동맥 혈관을 통해 **가이드와이어**를 넣어 관상동맥의 병변부위에 접근. **가이드와이어**를 통해 다른 시술도구들을 병변 부위까지 이동.
2. **벌룬**으로 막힌 혈관 부분을 넓히고, 병변에 따라 **스텐트**라는 그물망으로 고정

<center><img src="https://user-images.githubusercontent.com/17582508/67552494-3de99380-f746-11e9-8404-6491e42f1339.png" width="80%"></center>


이 때, 시술자는 조영제와 X-ray를 이용한 **혈관조영영상**을 이용해 병변을 확인하고 시술 위치를 파악합니다. 시술이 진행되는 도중에도 계속해서 혈관조영영상을 확인하여 정확한 시술을 진행합니다. 아래는 혈관조영영상의 일부입니다 [[7]](#ref_7).

<center><img src="https://user-images.githubusercontent.com/17582508/67552998-324a9c80-f747-11e9-8b0b-b5c979a6a9a7.png" width="80%"></center>

## 심혈관 중재시술 자동화의 필요성
---
PCI는 허혈성 심장 질환을 효과적으로 치료할 수 있는 시술 방법이지만 개선해야할 부분도 있습니다. 

1. 의료진의 방사능 피폭
    - PCI는 혈관의 모양과 현재 시술도구의 위치를 확인하기 위해 시술 중 혈관조영영상을 계속해서 촬영해야합니다. 촬영 기기 앞에 방사선 가림막을 두기는 하지만 매일 시술을 진행하는 의료진들은 많은 양의 방사능에 노출될 수 밖에 없습니다.

2. 긴 시술 시간
    - PCI는 일반적으로 1시간 정도의 시술시간이 소요됩니다. 하지만 병변 부위의 난이도에 따라 시술시간이 2시간에서 많게는 4시간 이상 걸리기도 합니다. 시술 시간이 길어질수록 시술자 뿐만 아니라 환자에게도 많은 부담이 됩니다. 환자의 경우 혈관조영영상 촬영시 환자의 혈관에 투입되는 조영제의 양이 늘어나게 되고 이는 가려움증, 구토와 같은 과민반응을 일으킬 수 있습니다.

3. 숙련도에 따른 차이
    - PCI는 시술자의 숙련도에 따라 시술 속도나 도구 사용량 등에서 편차가 있습니다. 특히 PCI의 시술 도구들은 1회용 소모품인 것에 비해 고가의 도구들이 많습니다. 숙련된 시술자라면 경험을 통해 병변 부위나 환자의 상태를 보고 한번에 적합한 시술 도구로 시술을 진행할 수 있지만 비숙련자는 여러 번의 시행착오를 통해 도구를 선택하게 됩니다. 이 때문에 자연스럽게 시술 시간이 길어지게 되어 환자나 시술자에게 부담을 주게 됩니다.

위와 같은 점들을 개선하기 위해 PCI 로봇에 대한 연구가 진행되고 있습니다. 대표적으로 Corindus의 시술 로봇이 있습니다 [[8]](#ref_8). 저희는 이보다 더 나아가 인공지능으로 로봇을 제어하여 시술을 **자동화**하는 연구를 진행하고 있습니다. PCI의 자동화를 통해 환자와 시술자 모두의 부담을 줄이게 되며, 특히 이상적으로 자동화가 된다면 시술의 숙련도에 따른 차이를 없앨 수 있습니다.

## 가이드와이어 제어
---
시술 자동화의 첫 단계로 **로봇을 이용한 가이드와이어의 자동제어**를 연구하고 있습니다. PCI는 위의 시술 과정에서 설명드린 것처럼 가이드와이어를 병변 부위까지 위치시킨 후 이 가이드와이어를 이용해 다른 시술 도구들을 이동시킵니다. 그렇기 때문에 가이드와이어를 병변 부위까지 잘 도달시키는 것이 시술의 가장 중요한 부분입니다. 하지만 혈관조영영상으로도 정확히 알 수 없는 **혈관의 구조와 모양**, 와이어 이동 시 생기는 **혈관에 의한 마찰**, 시술 중 환자의 **호흡과 박동**, 혈관에 들어가기 위해 얇고 부드럽게 제작되어 **제어가 어려운 와이어** 등 가이드와이어를 제어하는데 많은 어려움이 있습니다. 실제 의사들은 의료 지식을 통한 혈관 모양 예측, 손의 감각과 경험을 통한 미세한 조종으로 가이드와이어를 이동시켜 시술합니다. 저희는 가이드와이어 제어 문제를 **로봇과 인공지능**을 결합하여 해결하려 합니다. 

<center><img src="https://user-images.githubusercontent.com/17582508/67835790-3b17e580-fb2e-11e9-9f63-72a19a667a01.gif" width="80%"></center>

위 이미지 [[9]](#ref_9)에서 의사가 가이드와이어를 조작하는 모습과 로봇(Manipulator)를 이용해 가이드와이어를 조작하는 모습을 볼 수 있습니다. 실제 시술에서 시술자는 가이드와이어를 병변 부위로 이동시키기 위해 **회전(rotation)**과 **전후진(translation)** 동작을 합니다. 저희가 이용하는 로봇도 이러한 두 가지 동작을 재현할 수 있도록 설계되었습니다. 가이드와이어의 상태에 따라 두 가지 동작 중 어떤 동작을 할 지 선택하는 것이 인공지능의 역할입니다.

## 마치며
---
이번 포스트에서는 심혈관 중재시술과 저희가 진행하고 있는 연구 주제인 심혈관 중재시술 자동화의 필요성에 대해 설명드렸습니다. 다음 포스트에서는 PCI를 자동화하는 방법으로 인공지능 기법 중  강화학습을 선택한 이유에 대해서 포스팅하도록 하겠습니다.

[다음 포스트: 강화학습의 혈관 속 탐험 (2) - 강화학습과 제어 이론의 비교]({{< ref "2019-10-25-adventure_of_rl_in_vessel_2.md" >}})


## Reference
---
<a id="ref_1"></a>
**[1]** 심장 이미지 (wiki): https://ko.wikipedia.org/wiki/%EA%B4%80%EC%83%81%EB%8F%99%EB%A7%A5

<a id="ref_2"></a>
**[2]** 허혈성 심질환 설명(서울아산병원): http://www.amc.seoul.kr/asan/healthinfo/disease/diseaseDetail.do?contentId=30275

<a id="ref_3"></a>
**[3]** 연도별 국내 사망 원인 통계(e-나라지표), 세계 사망 원인(WHO): http://www.index.go.kr/potal/stts/idxMain/selectPoSttsIdxMainPrint.do?idx_cd=1012&board_cd=INDX_001, https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death

<a id="ref_4"></a>
**[4]** 심혈관질환의 종류 이미지(아스피린프로텍트): http://www.aspirinprotect.co.kr/ko/disease-and-prevention/cardiovascular-diseases/#tab_tab0

<a id="ref_5"></a>
**[5]** 관상동맥 중재시술의 이해(서울아산병원 심장병원): http://heart.amc.seoul.kr/asan/depts/heart/K/bbsDetail.do?menuId=4634&contentId=264501

<a id="ref_6"></a>
**[6]** 사람 몸과 혈관 이미지: http://www.secondscount.org/image.axd?id=c8a00122-bb66-46c6-8ab7-333a9a0cd46a&t=635566481777430000, https://www.mcvs.co.nz/wp-content/uploads/2017/05/stent-balloon-angioplasty.png

<a id="ref_7"></a>
**[7]** 관상동맥 조영술 이미지 (강남세브란스병원): http://gs.iseverance.com/dept_clinic/department/cardiology/treatment/view.asp?con_no=82261

<a id="ref_8"></a>
**[8]** Corindus: https://www.corindus.com/

<a id="ref_9"></a>
**[9]** 수기 시술과 로봇 이미지: 아산메디컬센터
