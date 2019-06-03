+++
title = "Histogram"
summary = "Histogram using OpenCV"
date = 2019-05-01T22:22:17+09:00
draft = false
authors=["young-kim", "whi-kwon"]
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

---
## Introduction

* OpenCV와 python으로 Image processing을 알아봅시다.
* 이 글은 첫 번째 posting으로, Image processing에서 Histogram이란 무엇인지에 대하여 알아보겠습니다.

---

## Import Libraries


```python
import os
import sys
import math
from platform import python_version

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


print("Python version : ", python_version())
print("OpenCV version : ", cv2.__version__)
matplotlib.rcParams['figure.figsize'] = (4.0, 4.0)
```

    Python version :  3.6.6
    OpenCV version :  3.4.3


## Data load


```python
sample_image_path = '../image/'
sample_image = 'lena_gray.jpg'
img = cv2.imread(sample_image_path + sample_image, cv2.IMREAD_GRAYSCALE)
```

## Data description
* 본 예제에서 사용할 데이터는 아래와 같습니다.
    * 거의 대부분의 OpenCV 예제에서 볼 수 있는 Lena 입니다.
    * 단순한 특징의 배경과 복잡한 특징의 인물이 함께 존재하여 다양한 condition을 test하기 좋은 data로 널리 알려져 있습니다.


```python
plt.imshow(img, cmap='gray')
plt.title('Lena')
plt.show()
```


{{< figure library="1" src="Histogram_7_0.png" >}}

---

## Histogram
* Histogram이란, 이미지에서 특정 픽셀값의 등장 빈도를 전체 픽셀 갯수 대비 비율로 나타낸 그래프 입니다.
* 이미지의 전체적인 명암 분포를 한 눈에 확인할 수 있습니다.
* 두 가지 예제 코드를 통해 Histogram에 대해 알아보겠습니다.


```python
def plot_histogram_npy(img):
    w, h = img.shape
    w, h = int(w), int(h)
    hist_cnt = np.zeros(255)
    hist = np.zeros(255)
    for j in range(h):
        for i in range(w):
            hist_cnt[img[j, i]] += 1
    hist = hist_cnt / (w * h)
    plt.plot(hist)
    plt.title('Histogram of Lena, numpy', size=15)
```


```python
plot_histogram_npy(img)
```

{{< figure library="1" src="Histogram_11_0.png" >}}



* OpenCV등을 이용하지 않고 numpy만을 이용하여 구한 Lena의 Histogram 입니다.
* 이미지에서 개별 픽셀값이 몇 번씩 등장하는지 확인하고 전체 픽셀 수로 normalize하여 Histogram을 얻게 됩니다.

---


```python
def plot_histogram_cv(img):
    w, h = img.shape
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_norm = hist / (w * h)  # normalize
    plt.plot(hist_norm)
    plt.title('Histogram of Lena, OpenCV', size=15)
```


```python
plot_histogram_cv(img)
```

{{< figure library="1" src="Histogram_15_0.png" >}}


* OpenCV를 이용하면 함수 호출을 통해 간단하게 Histogram을 구할 수 있습니다.
* numpy로 구한 Histogram과 비교해 보면, 두 결과물이 완전히 동일한 것을 알 수 있습니다.
* *'cv2.calcHist()'*를 수행하면 픽셀값 별 등장 횟수의 그래프를 얻고, 이를 normalize하여 최종적으로 Histogram을 얻게 됩니다.
* *'cv2.calcHist()'*의 자세한 사용법은 OpenCV 공식 tutorial page를 통해 확인할 수 있습니다. [[2]](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_begins/py_histogram_begins.html#histogram-calculation-in-opencv)
---

## Histogram&nbsp;Equalization
* Histogram Equalization(히스토그램 평활화)란, pixel값 0부터 255까지의 누적치가 직선 형태가 되도록 만드는 이미지 처리 기법 입니다.
    * 히스토그램 평활화 기법은 이미지가 전체적으로 골고루 어둡거나 골고루 밝아서 특징을 분간하기 어려울 때 자주 쓰입니다.
* 설명이 난해하니 코드를 통해 자세히 알아보겠습니다.


```python
def show_stacked_histogram(img):
    stack_hist = np.zeros(255, dtype=np.float32)
    eq_hist = np.zeros(255, dtype=np.float32)
    w, h = img.shape
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    for i in range(255):
        stack_hist[i] = np.sum(hist[:i])
        eq_hist[i] = round(stack_hist[i])
    eq_hist /= (w * h)

    plt.plot(eq_hist)
    plt.title('Stacked Histogram', size=15)
    plt.show()
```


```python
show_stacked_histogram(img)
```

{{< figure library="1" src="Histogram_19_0.png" >}}



* 0부터 255까지의 Histogram의 누적치 입니다. 쉽게 말하면 Histogram을 적분한 것이라고 할 수 있습니다.
    * e.g) eq_hist[150] = 0.675 &rarr; 이미지 내에서 __0부터 150까지의 pixel이 차지하는 비율 = 67.5%__
    * 당연히 항상 eq_hist[0] = 0이며, eq_hist[255] = 1.0 입니다.
* 전체적으로 직선에 가까운 형태지만 x좌표기준 0 근처와 255 근처는 수평인 것을 알 수 있습니다.
    * 이 말은 Lena image에서 pixel 값 기준 0 근처와 255 근처가 존재하지 않는다는 말 입니다.
    * 즉, 다시 말해 전체 pixel 값들에 대하여 분포가 균일하지 않다는 말 입니다.
* __히스토그램 평활화__는 이와 같이 __균일하지 않은 픽셀값의 분포를 고르게__ 만드는 작업입니다.
---
* 그럼 이제 히스토그램 평활화를 해보겠습니다.


```python
equ = cv2.equalizeHist(img)
show_stacked_histogram(equ)
```

{{< figure library="1" src="Histogram_21_0.png" >}}



* OpenCV에 구현되어있는 cv2.equalizeHist()함수를 통해 평활화한 Lena의 Histogram입니다.
* 이제 Histogram 누적치가 직선 형태라는 말이 확실하게 이해 되실 것 같습니다.
---
* 마지막으로 이렇게 변화시킨 이미지가 원본 이미지와 어떻게 다른지 확인해 보겠습니다.


```python
plt.figure(figsize=(8,8))

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original Lena')

plt.subplot(122)
plt.imshow(equ, cmap='gray')
plt.title('Equalized Lena')

plt.show()
```

{{< figure library="1" src="Histogram_23_0.png" >}}



* 전체적인 톤의 변화를 확인할 수 있는데, 밝은 부분은 더 밝아지고 어두운 부분은 더 어두워지는 모습을 볼 수 있습니다.
    * 이는 원본 이미지가 중간 정도 밝기의 픽셀을 다수 포함하고 있었고, 상대적으로 아주 어둡거나 아주 밝은 부분은 적었기 때문입니다.
    * 히스토그램 평활화를 통해 모든 픽셀값이 동일한 비율로 등장하게끔 수정하여 이와 같은 변화가 일어났다고 볼 수 있습니다.
* 히스토그램 평활화에 대한 자세한 내용은 마찬가지로 OpenCV 공식 tutorial page를 통해 확인할 수 있습니다. [[3]](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html#histogram-equalization)

********

## Conclusion

* 이미지 분석의 기초인 히스토그램에 대하여 알아보았습니다.
* 또한 이미지 처리 기법중 하나인 히스토그램 평활화를 알아보았으며, 실질적으로 히스토그램에 어떤 변화를 주는지 확인할 수 있었습니다.

---

## Reference
- [1] 오일석, 컴퓨터 비전, 2014, pp. 58-63
- [2] Alexander Mordvintsev & Abid K., 'Histograms - 1 : Find, Plot, Analyze !!!',  OpenCV-python tutorials. 2013 [Online]. Available: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_begins/py_histogram_begins.html#histogram-calculation-in-opencv [Accessed: 29- Mar- 2019]
- [3] Alexander Mordvintsev & Abid K., 'Histograms - 2: Histogram Equalization',  OpenCV-python tutorials. 2013 [Online]. Available: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html#histogram-equalization [Accessed: 29- Mar- 2019]
