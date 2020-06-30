
+++
title = "[Tutorial] Morphology"
summary = "Morphology Operations using OpenCV"
date = 2019-12-16T14:00:00+09:00
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
## Morphology
---
* 이번 장 에서는 Morphology Operation[[2]](https://en.wikipedia.org/wiki/Mathematical_morphology)에 대하여 알아보겠습니다.
    * Morphology Operation이란, Image의 형태를 조작하는 연산의 종류를 통칭합니다.
    * 적용 대상이 어떤 Image냐에 따라 Binary, Grayscale로 나눌 수 있고, 연산 방식에 따라 Dilation(확산)과 Erosion(침식)으로 나눌 수 있습니다.
    * 두 연산 모두 공통적으로 Source Image Kernel을 통해 Target Image를 생성하는 과정을 갖습니다.
    * Morphology Operation을 응용하면 Edge Detection이나 Image Denoising 등의 작업을 할 수 있습니다.
* 목차는 아래와 같습니다.
- [Binary Morphology](#binarynbspmorphology)
- [Grayscale Morphology](#grayscalenbspmorphology)
- [Composited Morphological Operations](#compositednbspmorphologicalnbspoperations)


### Import Libraries


```python
import os
import sys
import math
from platform import python_version

import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

print("Python version : ", python_version())
print("Opencv version : ", cv2.__version__)
```

    Python version :  3.6.9
    Opencv version :  4.1.2


### Data load


```python
sample_image_path = '../image/'
sample_image = 'kitten.jpg'
img = cv2.imread(os.path.join(sample_image_path, sample_image), cv2.IMREAD_GRAYSCALE)
h, w = [int(x) for x in img.shape]
matplotlib.rcParams['figure.figsize'] = (8.0, 8.0)
```


## Binary&nbsp;Morphology
---

* 전체 Image가 0 또는 1로 이루어진 Image를 Binary Image라고 합니다. 아래 설명은 모두 Binary Image를 기준으로 하겠습니다.
* 이러한 Image는 보통 Image 자체로 이용되기보다 RoI(Region of Interest)를 선택하기 위한 Mask로서 이용됩니다.
* Binary Image에 Morphology를 적용하면 자잘한 Noise를 제거하는데 도움이 됩니다.
* Erosion 연산은 침식이라는 말 뜻이 의미하듯이 1에 해당하는 밝은 영역이 조금씩 줄어드는 모습을 보입니다.
    * RoI 바깥에 1의 값을 갖는 Noise가 발생하는 경우에 Erosion 연산을 통해 Noise를 보정할 수 있습니다.
* Dilation 연산은 팽창이라는 뜻인데, 흰색 영역이 조금 커지는 모습을 보입니다.
    * RoI 중간에 0의 값을 갖는 Noise가 발생하는 경우에 Dilation 연산으로 보정할 수 있습니다.
* 한 번의 호출로 몇 번씩 연산을 반복할지를 인자 'iterations'로 조절할 수 있습니다.


```python
ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
erosion = cv2.erode(th, kernel, iterations=1)
dilation = cv2.dilate(th, kernel, iterations=1)

th = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
erosion = cv2.cvtColor(erosion, cv2.COLOR_GRAY2RGB)
dilation = cv2.cvtColor(dilation, cv2.COLOR_GRAY2RGB)

th = cv2.rectangle(th, (200, 140), (400, 340), (255, 0, 0), 3)
erosion = cv2.rectangle(erosion, (200, 140), (400, 340), (255, 0, 0), 3)
dilation = cv2.rectangle(dilation, (200, 140), (400, 340), (255, 0, 0), 3)

plt.subplot(221)
plt.imshow(th, cmap='gray')
plt.title('Binary Kitten')

plt.subplot(223)
plt.imshow(erosion, cmap='gray')
plt.title('Erosed Kitten')

plt.subplot(224)
plt.imshow(dilation, cmap='gray')
plt.title('Dilated Kitten')

plt.suptitle('Binary Kitten with morphology', size=15)
plt.show()
```

{{< figure library="1" src="opencv/2.6/2019-12-16-morphology_6_1.png" >}}


## Grayscale&nbsp;Morphology
---
* Morphology Operation을 Grayscale Image에 적용하게 되면 기본 원리는 동일하지만 결과가 아주 달라집니다.
* 1, 0의 값을 사용하는 것이 아니라 Dilation의 경우 Max값을, Erosion의 경우 Min 값을 사용한다는 점이 다른 점 입니다.
* 자세한 것은 아래 수식을 통해 확인할 수 있습니다.
    * grayscale dilation : $ (I\oplus k)(j,i)=max_{(y,x)\subseteq k} (I(j-y,i-x)+k(y,x))$
    * grayscale erosion : $ (I\ominus k)(j,i)=min_{(y,x)\subseteq k} (I(j+y,i+x)-k(y,x))$
        * I = Image, k = kernel
* 수식에서 알 수 있듯 Min, Max연산이 적용됩니다. Dilation의 경우 큰 값을 더 크게, 작은 값을 더 작게 하여 픽셀값간의 차이를 극대화하는 연산이고, Erosion의 경우 큰 값을 작게, 작은 값을 크게 하여 전체적으로 평평하게 만드는 연산입니다.


```python
kernel = np.array([[1, 2, 1]])
gray_kernel = np.dot(kernel.T, kernel)
gray_erosion = cv2.erode(img, gray_kernel, iterations=1)
gray_dilation = cv2.dilate(img, gray_kernel, iterations=1)

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Gray Kitten')

plt.subplot(223)
plt.imshow(gray_erosion, cmap='gray')
plt.title('Gray Erosed Kitten')

plt.subplot(224)
plt.imshow(gray_dilation, cmap='gray')
plt.title('Gray Dilated Kitten')

plt.suptitle('Gray Kitten with morphology', size=15)
plt.show()
```


{{< figure library="1" src="opencv/2.6/2019-12-16-morphology_9_0.png" >}}


* 어떤 Image에 커널을 적용하느냐가 Morphology Operation 의 핵심입니다. 물론 어떤 커널을 설계하느냐도 중요하지만, 어떤 연산을 어떤 순서로 몇 번 씩 적용할 것인가 또한 결과물에 많은 영향을 끼칩니다.
    * 위 결과물을 얻는데 사용한 kernel은 가운데쪽에 더 많은 가중치를 주게 되어 주변과의 명암 차이를 더 크게 하는 효과가 있습니다.
* Morphology연산은 Heuristic한 특성이 크게 드러나는 연산입니다.
    * 다양한 특성이 존재하는 Image 전체에 일괄적으로 동일한 Morphology Operation을 적용하는 것은 대부분 의미가 없고, 동일한 특성을 가진 부분에 특정한 목적 위해 적용하는 편이 좋습니다.
    * 이러한 작업을 잘 하기 위해서는 결국 다양한 Image에 다양한 방법으로 다양한 커널을 적용해보면서 감을 잡아야 합니다.


## Composited&nbsp;Morphological&nbsp;Operations
---
* 앞서 소개한 두 Morphology Operation을 번갈아 한 번 씩 적용하는 작업 또한 상당히 빈번하게 사용됩니다.
    * 한 번 변형한 Mask를 원래 크기로 되돌리기 위해서인데, 이러한 작업을 통해 원하는 노이즈만 제거된 결과를 얻을 수 있습니다.
* 작업을 진행하는 순서에 따라 아래 두 가지로 나눌 수 있습니다.
    * Erosion&rarr;Dilation의 경우 Open(열기)
    * Dilation의&rarr;Erosion 경우 Close(닫기)
* Open은 1 값을 갖는 노이즈를 줄이고, Close는 0값을 갖는 노이즈를 줄이는데 사용됩니다.


```python
ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

erosion = cv2.erode(th, kernel, iterations=1)
dilation = cv2.dilate(th, kernel, iterations=1)
opened = cv2.dilate(erosion, kernel, iterations=1)
closed =  cv2.erode(dilation, kernel, iterations=1)

plt.subplot(221)
plt.imshow(th, cmap='gray')
plt.title('Binary Kitten')

plt.subplot(223)
plt.imshow(opened, cmap='gray')
plt.title('Opened Kitten')

plt.subplot(224)
plt.imshow(closed, cmap='gray')
plt.title('Closed Kitten')

plt.suptitle('Binary Kitten with double morphology', size=15)
plt.show()
```


{{< figure library="1" src="opencv/2.6/2019-12-16-morphology_12_0.png" >}}


* Gray Scale Image에도 동일한 작업을 진행할 수 있습니다.


```python
gray_open = cv2.dilate(gray_erosion, kernel, iterations=2)
gray_close =  cv2.erode(gray_dilation, kernel, iterations=2)

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Gray Kitten')

plt.subplot(223)
plt.imshow(gray_open, cmap='gray')
plt.title('Gray opened Kitten')

plt.subplot(224)
plt.imshow(gray_close, cmap='gray')
plt.title('Gray closed Kitten')

plt.suptitle('Gray Kitten with double morphology', size=15)
plt.show()
```


{{< figure library="1" src="opencv/2.6/2019-12-16-morphology_14_0.png" >}}

## Conclusion
---
* Morphology Operation은 주로 Binary Image로 생성한 RoI Mask를 다루는데 많이 쓰입니다.
* 개인적으로 RoI Mask의 Denoising에 매우 유용하게 사용하고 있고, Edge Detection 등의 연산과 연계하는 등 활용 방안이 아주 많으니 꼭 익혀두시는 것을 추천드립니다.

## Reference
---


* [1] 오일석, "모폴로지," in 컴퓨터 비전, vol.4, Republic of Korea:한빛아카데미, 2014, pp. 97-103
* [2] Accessed: 'Mathematical morphology', Wikipedia. [Online]. Available: https://en.wikipedia.org/wiki/Mathematical_morphology
