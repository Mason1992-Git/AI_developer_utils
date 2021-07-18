import numpy
import cv2
from PIL import Image
import torchvision

"""
opencv转PIL
"""

img = cv2.imread("images/1.jpg")
#判断是否为cv2
print(isinstance(img, numpy.ndarray))
# cv2.imshow("image",img)
# cv2.waitKey()
img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
print(isinstance(img, numpy.ndarray))
# img.show()

"""
PIL转opencv
"""

img = cv2.cvtColor(numpy.asarray(img),cv2.COLOR_RGB2BGR)
print(isinstance(img, numpy.ndarray))
# cv2.imshow("img",img)
# cv2.waitKey()

"""
torchvision.transforms.ToPILImage()

将Numpy的ndarray或者Tensor转化成PILImage类型【在数据类型上，两者都有明确的要求】
ndarray的数据类型要求dtype=uint8, range[0, 255] and shape H x W x C
Tensor 的shape为 C x H x W 要求是FloadTensor的，不允许DoubleTensor或者其他类型
"""
img = torchvision.transforms.ToPILImage()(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
print(isinstance(img, numpy.ndarray))
img.show()