# ADAFMNoiseReducer
![Image description](imgs/img.jpg)


This is a fork from https://github.com/hejingwenhejingwen/AdaFM
The code has been refactored for a better readability and simplified for inference only.

## Architecture 
```
AdaResNet(
  (model): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): Identity + 
    |Sequential(
    |  (0): ResNetBlock(
    |    (res): Sequential(
    |      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (1): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |      (2): ReLU(inplace=True)
    |      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (4): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |    )
    |  )
    |  (1): ResNetBlock(
    |    (res): Sequential(
    |      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (1): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |      (2): ReLU(inplace=True)
    |      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (4): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |    )
    |  )
    |  (2): ResNetBlock(
    |    (res): Sequential(
    |      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (1): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |      (2): ReLU(inplace=True)
    |      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (4): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |    )
    |  )
    |  (3): ResNetBlock(
    |    (res): Sequential(
    |      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (1): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |      (2): ReLU(inplace=True)
    |      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (4): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |    )
    |  )
    |  (4): ResNetBlock(
    |    (res): Sequential(
    |      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (1): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |      (2): ReLU(inplace=True)
    |      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (4): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |    )
    |  )
    |  (5): ResNetBlock(
    |    (res): Sequential(
    |      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (1): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |      (2): ReLU(inplace=True)
    |      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (4): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |    )
    |  )
    |  (6): ResNetBlock(
    |    (res): Sequential(
    |      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (1): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |      (2): ReLU(inplace=True)
    |      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (4): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |    )
    |  )
    |  (7): ResNetBlock(
    |    (res): Sequential(
    |      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (1): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |      (2): ReLU(inplace=True)
    |      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (4): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |    )
    |  )
    |  (8): ResNetBlock(
    |    (res): Sequential(
    |      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (1): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |      (2): ReLU(inplace=True)
    |      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (4): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |    )
    |  )
    |  (9): ResNetBlock(
    |    (res): Sequential(
    |      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (1): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |      (2): ReLU(inplace=True)
    |      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (4): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |    )
    |  )
    |  (10): ResNetBlock(
    |    (res): Sequential(
    |      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (1): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |      (2): ReLU(inplace=True)
    |      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (4): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |    )
    |  )
    |  (11): ResNetBlock(
    |    (res): Sequential(
    |      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (1): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |      (2): ReLU(inplace=True)
    |      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (4): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |    )
    |  )
    |  (12): ResNetBlock(
    |    (res): Sequential(
    |      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (1): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |      (2): ReLU(inplace=True)
    |      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (4): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |    )
    |  )
    |  (13): ResNetBlock(
    |    (res): Sequential(
    |      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (1): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |      (2): ReLU(inplace=True)
    |      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (4): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |    )
    |  )
    |  (14): ResNetBlock(
    |    (res): Sequential(
    |      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (1): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |      (2): ReLU(inplace=True)
    |      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (4): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |    )
    |  )
    |  (15): ResNetBlock(
    |    (res): Sequential(
    |      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (1): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |      (2): ReLU(inplace=True)
    |      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |      (4): AdaptiveFM(
    |        (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |      )
    |    )
    |  )
    |  (16): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    |  (17): AdaptiveFM(
    |    (transformer): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64)
    |  )
    |)
    (2): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): PixelShuffle(upscale_factor=2)
    (4): ReLU(inplace=True)
    (5): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
)
```

## Installation
```
pip install -r adafmnoisereducer-1.0-py36-none-any.whl
```
## Getting started
Test an inference on sample image
```python
from ADAFMNoiseReducer import ADAFMNoiseReducer
import cv2

if __name__ == "__main__":
    reducer = ADAFMNoiseReducer()
    img_path = "input.jpg"
    img = cv2.imread(img_path)
    img = reducer(img)
    cv2.imwrite("result.jpg", img)
    cv2.imshow("", img)
    cv2.waitKey()

```
