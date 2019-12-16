
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
# Morphology
* 이번 장 에서는 Morphology operation에 대하여 알아보겠습니다.[1](#ref_1),[2](#ref_2)
    * Morphology operation이란, 이미지의 형태를 조작하는 연산의 종류를 통칭합니다.
    * 적용 대상이 어떤 이미지냐에 따라 binary, grayscale로 나눌 수 있고, 연산 방식에 따라 dilation(확산)과 erosion(침식)으로 나눌 수 있습니다.
    * 위 분류와 관계 없이 공통적으로 대상 이미지와 kernel을 통해 target image를 생성하는 과정을 거칩니다.
* 목차는 아래와 같습니다.
- [Binary morphology](#ref_3)
- [Grayscale morphology](#ref_4)
- [Composited Morphological operations](#ref_5)
---

## Import Libraries


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


## Data load


```python
sample_image_path = '../image/'
sample_image = 'kitten.jpg'
img = cv2.imread(sample_image_path + sample_image, cv2.IMREAD_GRAYSCALE)
h, w = [int(x) for x in img.shape]
matplotlib.rcParams['figure.figsize'] = (8.0, 8.0)
```

---

<a id="ref_3"></a>

## Binary&nbsp;morphology

* 전체 이미지가 0 또는 1로 이루어진 이미지를 Binary 이미지라고 합니다.
* 이러한 이미지는 보통 이미지 자체로 이용되기보다 RoI(region of interest)를 선택하기 위한 mask로서 이용됩니다.
* Binary image에 morphology를 적용하면 자잘한 Noise를 제거하는데 도움이 됩니다.

* erode 연산은 침식이라는 말 뜻이 의미하듯이 1에 해당하는 밝은 영역이 조금씩 줄어드는 모습을 보입니다.
    * 관심 없는 영역에 mask가 1의 값을 자잘하게 갖는 경우에 erode연산을 통해 Noise를 보정할 수 있습니다.
* dilate 연산은 팽창이라는 뜻인데, 흰색 영역이 조금 커지는 모습을 보입니다.
    * RoI 중간에 자잘하게 뚫린 영역을 dilate연산으로 보정할 수 있습니다.
* 만족스러운 품질을 얻을 때 까지 iterations를 높여가며 반복할 수 있습니다.


```python
ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
erosion = cv2.erode(th, kernel, iterations=1)
dilation = cv2.dilate(th, kernel, iterations=1)

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

{{< figure library="1" src="opencv/2.6/2019-12-16-morphology_6_0.png" >}}

<a id="ref_4"></a>

## Grayscale&nbsp;morphology
* 모폴로지 연산을 grayscale 영상에 적용하게 되면 기본 원리는 동일하지만 결과가 아주 달라집니다.
* 1, 0의 값을 사용하는 것이 아니라 dilation의 경우 max값을, erosion의 경우 min 값을 사용한다는 점이 다른 점 입니다.
* 자세한 것은 아래 수식을 통해 확인할 수 있습니다.

* grayscale dilation : $ (f\oplus S)(j,i)=max_{(y,x)\subseteq S} (f(j-y,i-x)+S(y,x))$

* grayscale erosion : $ (f\ominus S)(j,i)=min_{(y,x)\subseteq S} (f(j+y,i+x)-S(y,x))$

* 수식에서 알 수 있듯 min, max연산이 적용됩니다. dilation의 경우 큰 값을 더 크게, 작은 값을 더 작게 하여 픽셀값간의 차이를 극대화하는 연산이고, erosion의 경우 큰 값을 작게, 작은 값을 크게 하여 전체적으로 평평하게 만드는 연산입니다.


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


* 어떤 이미지에 커널을 적용하느냐가 모폴로지 연산 성능의 핵심입니다.
    * 물론 어떤 커널을 설계하느냐도 중요하지만, 어떤 연산을 어떤 순서로 몇 번 씩 적용할 것인가 또한 결과물에 지대한 영향을 끼칩니다.
* morphology연산 heuristic한 특성이 크게 드러나는 연산입니다.
    * 다양한 특성이 존재하는 이미지 전체에 일괄적으로 동일한 모폴로지 연산을 적용하는 것은 의미가 없고 동일한 특성이 적용되는 부분에 적용하여 특정 목적을 달성하기 위해 사용하는 편이 좋습니다.
    * 그러한 작업을 잘 하기 위해서는 결국 다양한 이미지에 다양한 방법으로 다양한 커널을 적용해보면서 감을 잡아야 합니다.

<a id="ref_5"></a>

## Composited&nbsp;Morphological&nbsp;operations
* 앞서 소개한 두 Morphology operation 연산을 번갈아 한 번 씩 적용하는 작업 또한 상당히 빈번하게 사용됩니다.
    * 한 번 변형한 mask를 원래 크기로 되돌리기 위해서인데, 이러한 작업을 통해 원하는 노이즈만 제거된 결과를 얻을 수 있습니다.
* 작업을 진행하는 순서에 따라 아래 두 가지로 나눌 수 있습니다.
    * erosion->dilation의 경우 open(열기)
    * dilation의->erosion 경우 close(닫기)
* open은 1 값을 갖는 노이즈를 줄이고, close는 0값을 갖는 노이즈를 줄이는데 사용됩니다.


```python
ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

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


* gray scale image에도 동일한 작업을 진행할 수 있습니다.


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


---
## Reference
<a id="ref_2"></a>

* [1] 오일석, 컴퓨터 비전, 2014, pp. 97-103
<a id="ref_2"></a>

* [2] 'Morphological Image Processing', Auckland University. [Online]Available: https://www.cs.auckland.ac.nz/courses/compsci773s1c/lectures/ImageProcessing-html/topic4.htm
