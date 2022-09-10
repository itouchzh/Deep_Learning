import imp
import qrcode
import cv2
from PIL import Image

data = '馨馨哇，这个东西好像不是那么好玩'
# img_data = cv2.imread('../../data/images/2.jpg')
# img_data = Image.open('../../data/images/2.jpg')
# print(img_data.shape)
img = qrcode.make(data)
img.show()
# img.save('../data/hello.png')
