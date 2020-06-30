
+++
title = "[Tutorial] Multi-scale"
summary = "Handling multi-scale images using OpenCV"
date = 2020-06-30T10:20:00+09:00
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
# Multi-scale
---
* 이번 장 에서는 다해상도(Multi-scale)에 대한 개념과 전통적인 Computer Vision에서 Multi-scale Image 처리를 하는 방법을 알아보겠습니다.
* 일반적으로 하나의 해상도에 최적화된 Computer Vision Model보다 다해상도를 처리할 수 있는 Computer Vision Model이 더 좋은 성능을 낼 가능성이 높기 때문에 Multi-scale 문제는 꼭 해결해야만 하는 문제입니다.
* 여기에서는 가장 단순하고 오래된 기법인 Image Pyramid기법만을 알아보겠습니다. 단순하게 요약하면 피라미드로 쌓은 것 같이 다양한 크기의 Image를 많이 준비하는 방법 입니다.
    - [Image pyramid](#imagenbspPyramid)


### Import Libraries


```python
import os
import sys
import math
from platform import python_version

import cv2
import matplotlib.pyplot as plt
import matplotlib
import imutils
import numpy as np

print(f"Python version : {python_version()}", )
print(f"Opencv version : {cv2.__version__}", )
```

    Python version : 3.6.9
    Opencv version : 4.1.2


### Data load


```python
sample_image_path = '../image/'
sample_image = 'kitten.jpg'
img = cv2.imread(os.path.join(sample_image_path, sample_image), cv2.IMREAD_GRAYSCALE)
h, w = img.shape
matplotlib.rcParams['figure.figsize'] = (8.0, 8.0)
```

## Image&nbsp;Pyramid
---

* 전통적인 Computer Vision이나 Deep Learning 모두 공통적으로 크기 변화에 취약하다는 단점을 가지고 있습니다.
    * 즉, 같은 모델로 더 크거나 작은 Image를 분석할 경우 결과가 전혀 달라질 수 있다는 말 입니다.
* 이에 대응하기 위해 하여 Feature추출, 추론 등에서 다양한 해상도의 Image를 사용하는 것이 보통입니다.
* Image Pyramid[[2]](https://docs.opencv.org/3.4/dc/dff/tutorial_py_pyramids.html)란, Image를 다양한 해상도의 Image로 변환하고 쌓아올려서 마치 피라미드와 같은 형상을 띄도록 준비하는 전처리 기법입니다.
* 이는 Multi-scale 문제를 해결해야 할 때 자주 쓰이던 기법이며, Deep Learning에서 쓰이는 FPN 등의 방법들이 이 Image Pyramid 기법을 개선하는 과정에서 고안된 것임을 감안할 때, Image Pyramid기법은 Multi-scale 문제 해결의 시작점 이라고 할 수 있습니다.


### 다해상도 처리
* Image Pyramid에서 결국 관건은 다양한 해상도의 Image를 어떻게 준비할 것이냐 인데, 크기가 작은 Image라면 Upsampling을 해서 더 큰 해상도로 만들어야 하고, 크기가 큰 Image라면 Downsampling을 해서 Image 크기를 줄여야 합니다.
* 그러나 두 방법 모두 단순히 적용하면 정보량의 변화로 인해 원래 Image보다 다소 부자연스러워 보이게 되는 Aliasing이라는 문제가 발생하게 되는데, 이를 보완하는 방법에 대해 간단히 알아보겠습니다.

### Upsample
* Upsample에서 발생하는 Aliasing문제는 대부분의 경우 지난 장에서 확인한 Interpolation 방법으로 해결합니다.
* 생성모델이나 한층 더 높은 정교함을 요구하는 모델의경우 데이터에 맞게 학습된 Filter를 이용하는 Deconvolution 등이 사용되기도 합니다.

### Downsample

* Downsample의 경우 특정 화소를 선택적으로 제거해야 하므로 문제가 발생합니다.
* 단순히 짝수번째나 홀수번째 화소를 제거하는 방식을 사용할 경우, 제거되는 화소만큼 그대로 손실이 되기 때문에 Image의 품질이 빠르게 나빠지는 것을 확인할 수 있습니다.


```python
img_half = cv2.resize(img, (w // 2, h // 2))
img_quard = cv2.resize(img, (w // 4, h // 4))
img_eight = cv2.resize(img, (w // 8, h // 8))

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Original image')

plt.subplot(222)
plt.imshow(img_half, cmap='gray')
plt.title('1/2 sized')

plt.subplot(223)
plt.imshow(img_quard, cmap='gray')
plt.title('1/4 sized')

plt.subplot(224)
plt.imshow(img_eight, cmap='gray')
plt.title('8/1 sized')

plt.suptitle('Kitten with many size', size=15)
plt.show()
```

{{< figure library="1" src="opencv/2.5/2019-12-16-multi_scale_10_0.png" >}}



* 이러한 문제를 해결하기 위한 대안은 원본 Image에 Smoothing을 적용한 후 Downsampling을 적용하는 것 입니다. 선택적으로 제거되는 픽셀의 정보를 주변 픽셀들에 조금씩 반영해주는 원리입니다.
* 아래 예제는 교재에서 사용된 커널 함수로 Smoothing을 적용한 결과입니다.


```python
kernel = np.array([[0.05, 0.25, 0.4, 0.25, 0.05]])
kernel = np.dot(kernel.T, kernel)

blur_img_half = cv2.resize(cv2.filter2D(img, -1, kernel), (w // 2, h // 2))
blur_img_quard = cv2.resize(
    cv2.filter2D(blur_img_half, -1, kernel), (w // 4, h // 4))
blur_img_eight = cv2.resize(
    cv2.filter2D(blur_img_quard, -1, kernel), (w // 8, h // 8))

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(222)
plt.imshow(blur_img_half, cmap='gray')
plt.title('Blur_half')

plt.subplot(223)
plt.imshow(blur_img_quard, cmap='gray')
plt.title('Blur_quard')

plt.subplot(224)
plt.imshow(blur_img_eight, cmap='gray')
plt.title('Blur_eight')

plt.suptitle('Soft Kitten with many size', size=15)
plt.show()
```


{{< figure library="1" src="opencv/2.5/2019-12-16-multi_scale_12_0.png" >}}

## Conclusion
---
* Multi-scale Image를 생성할 때 발생하는 문제와 이에 대응하기 위해 필요한 작업들을 살펴보았습니다.
* 여기에서 Smoothing 작업을 진행할 때, 다양한 커널을 이용할 수 있는데, Image에 따라 목적에 따라 최적의 커널이 따로 존재합니다.
    * 이 말은 그때 그때 알맞은 커널을 직접 찾아서 사용해야 한다는 뜻이기도 합니다. 이러한 부분이 전통적인 방식의 Computer Vision이 가지고 있었던 고질적인 약점입니다.

## Reference
---
* [1] 오일석, "다해상도," in 컴퓨터 비전, vol.4, Republic of Korea:한빛아카데미, 2014, pp. 93-96
* [2] Accessed: 'Image Pyramids', opencv official, [Online] Avaliable: https://docs.opencv.org/3.4/dc/dff/tutorial_py_pyramids.html