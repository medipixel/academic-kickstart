
+++
title = "[Tutorial] Multi Scale"
summary = "Handling multi scale images using OpenCV"
date = 2019-12-16T13:00:00+09:00
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
# Multi scale
* 이번 장 에서는 다해상도(Multi scale)에 대한 개념과 전통적인 computer vision에서 다해상도 이미지 처리를 하는 방법을 알아보겠습니다.
* 같은 feature를 사용하더라도 서로 다른 해상도의 이미지에서 여러번 얻는다면 좋은 모델을 구성하는데 한 걸음 가까워질 수 있습니다.
* 여기에서는 가장 단순하고 오래된 기법인 Image pyramid기법만을 알아보겠습니다.
    - [Image pyramid](#ref_1)
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
import imutils
import numpy as np

print(f"Python version : {python_version()}", )
print(f"Opencv version : {cv2.__version__}", )
```

    Python version : 3.6.9
    Opencv version : 4.1.2


## Data load


```python
sample_image_path = '../image/'
sample_image = 'kitten.jpg'
img = cv2.imread(sample_image_path + sample_image, cv2.IMREAD_GRAYSCALE)
h, w = img.shape
matplotlib.rcParams['figure.figsize'] = (8.0, 8.0)
```

---
## Image&nbsp;pyramid
* 전통적인 computer vision이나 deep learning을 사용한 방식 모두 공통적으로 scale 변화에 취약하다는 단점이 있습니다.
    * 다시 말해, 같은 이미지를 두 배 만큼 키우거나 줄여서 동일한 모델을 통과시킬 경우 결과값이 전혀 달라질 수 있다는 말 입니다.
* 이에 대응하기 위해 하여 feature추출, 추론 등에서 다양한 해상도의 이미지를 사용하는 것이 보통입니다.
* Image pyramid란, 동일한 이미지에 대하여 다양한 해상도의 이미지로 쌓아올려서 마치 피라미드와 같은 형상을 띄도록 준비 하는 전처리 기법입니다. [2](#ref_2)
* 이는 multi scale 문제를 해결해야 할 때 자주 쓰이던 기법이며, deep learning에서 쓰이는 FPN 등의 방법들이 이 image pyramid 기법을 개선하는 과정에서 고안된 것임을 감안할 때, Image pyramid기법은 multi scale 문제 해결의 시작점 이라고 할 수 있습니다. [3](#ref_3)


### 다해상도 처리
* Image pyramid에서 결국 관건은 다양한 해상도의 이미지를 어떻게 준비할 것이냐 인데, 크기가 작은 이미지라면 Upsampling을 해서 더 큰 해상도로 만들어야 할 것이고, 크기가 큰 이미지라면 Downsampling을 해야 할 것입니다.
* 다양한 해상도의 이미지를 준비하기 위하여 가장 쉽게 생각할 수 있는 방법으로 upsample과 downsample이라는 기법이 있습니다.
* 그러나 두 방법 모두 단순히 적용하면 정보량의 변화로 인해 원래 이미지보다 다소 부자연스러워 보이게 되는 aliasing이라는 문제가 발생하게 되는데, 이를 보완하는 방법에 대해 간단히 알아보겠습니다.

### Upsample
* Upsample에서 발생하는 aliasing문제는 대부분의 경우 지난 장에서 확인한 interpolation 방법으로 해결합니다.
* 생성모델이나 한층 더 높은 정교함을 요구하는 모델의경우 데이터에 맞게 학습된 filter를 이용하는 deconvolution 등이 사용되기도 합니다.

### Downsample

* downsample의 경우 특정 화소를 선택적으로 제거해야 하므로 문제가 발생합니다.
* 단순히 짝수번째나 홀수번째 화소를 제거하는 방식을 사용할 경우, 제거되는 화소만큼 그대로 손실이 되기 때문에 이미지의 품질이 빠르게 나빠지는 것을 확인할 수 있습니다.


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


---
* 이러한 문제를 해결하기 위한 대안은 원본 이미지에 smoothing을 적용한 후 downsampling을 적용하는 것 입니다. 선택적으로 제거되는 픽셀의 정보를 주변 픽셀들에 조금씩 반영해주는 원리입니다.
* 여기에서 smoothing 작업을 진행할 때, 다양한 커널을 이용할 수 있습니다.
* 이번 예제에서 사용한 커널의 경우 교재에 실려있는 커널 함수를 그대로 적용한 것인데[1], 목적에 따라서 더 좋은 커널이 있을 수 있다고 합니다.
    * 이 말은 목적에 따라 그때 그때 알맞은 커널을 직접 찾아서 사용해야 한다는 뜻이기도 합니다. 이러한 부분이 전통적인 방식의 computer vision이 가지고 있었던 고질적인 약점입니다.


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


---
### Reference
* [1] 오일석, 컴퓨터 비전, 2014, pp. 93-96
<a id="ref_2"></a>

* [2] 'Image Pyramids', opencv official, [Online] Avaliable: https://docs.opencv.org/3.4/dc/dff/tutorial_py_pyramids.html
<a id="ref_3"></a>

* [3] Adrian Rosebrock, Image Pyramids with Python and OpenCV, pyimagesearch . 2015 [Online] Available: https://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/
