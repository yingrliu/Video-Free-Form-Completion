"""
Author: Yingru Liu
This file contains the basic toolkit to generate sketch, mask and colormap of a given image.
"""

import cv2
import os
import torch
import numpy as np
from Toolkit.pytorch_hed.run import edge_detector
from Toolkit.CannyEdge.core import gradient_intensity, suppression
# import torch.nn.functional as F
# from sklearn.cluster import KMeans
# from torchvision import transforms
# from Toolkit.deeplabv3.modeling.deeplabv3p import DeepLabV3p
#


def sketchGenerator(img_path):
    """
    use the Canny Edge detector to generate image sketch.
    :param img_path: the path of RGB image.
    :return: sketch: the ndarray-form of the sketch.
    :return: sketch_tensor: the torch tensor form of the sketch. used for training or inference.
    """
    gs_img = edge_detector(img_path)
    #
    sketch = np.asarray(gs_img, dtype=np.float32)
    sketch = cv2.GaussianBlur(sketch, (3, 3), 10)
    # NMS
    _, D = gradient_intensity(sketch)
    sketch = suppression(sketch, D)
    # smooth.
    sketch = sketch.astype(dtype=np.uint8)
    sketch = cv2.GaussianBlur(sketch, (3, 3), 3)
    sketch[sketch >= 5] = 255
    sketch = cv2.medianBlur(sketch, 3)
    return sketch

def colorGenerator_by_Filter(img):
    """
    use the pretrained DeepLab-v3+ to extract the median color of each part of the person.
    The pytorch implementation of DeepLab-v3 from [https://github.com/jfzhang95/pytorch-deeplab-xception.git]
    is used.
    :param img: RGB image.
    :return: color: the ndarray-form of the color.
    :return: color_tensor: the torch tensor form of the color. used for training or inference.
    """
    #
    color = cv2.medianBlur(img, 3)
    for i in range(40):
        color = cv2.bilateralFilter(color, 9, 25, 25)
    # color = oilPainting(img, 4, 6, 2)
    return color


#######################################################################################################################
def maskGenerator(img, MaxThin=25, MaxDraw=15, MaxLine=50, MaxAngle=0.25*np.pi, MaxLength=50):
    h, w = img.shape[0], img.shape[1]
    mask = np.ones((h, w), dtype=np.float32)
    numDraw = np.random.randint(int(0.4 * MaxDraw), MaxDraw)
    for _ in range(numDraw):
        startX = np.random.randint(0, h)
        startY = np.random.randint(0, w)
        startAngle = np.random.uniform(0, 2 * np.pi)
        numLine = np.random.randint(int(0.4 * MaxLine), MaxLine)
        for j in range(numLine):
            angleP = np.random.uniform(-MaxAngle, MaxAngle)
            angle = startAngle + angleP if j % 2 == 0 else startAngle + angleP + np.pi
            length = np.random.uniform(0.4*MaxLength, MaxLength)
            thickness = np.random.randint(int(0.8 * MaxThin), MaxThin)
            endX = int(startX + length * np.sin(angle))
            endY = int(startY + length * np.cos(angle))
            cv2.line(mask, (startX, startY), (endX, endY), 0.0, thickness=thickness)
            startX, startY = endX, endY
    mask = np.expand_dims(mask, -1)
    return mask


#######################################################################################################################
def strokeGenerator(img, MaxThin=8, MaxDraw=75, MaxLine=5, MaxAngle=0.3*np.pi, MaxLength=50):
    h, w = img.shape[0], img.shape[1]
    stroke = np.zeros((h, w), dtype=np.float32)
    for _ in range(MaxDraw):
        startX = np.random.randint(0, h)
        startY = np.random.randint(0, w)
        startAngle = np.random.uniform(0, 2 * np.pi)
        numLine = np.random.randint(int(0.4 * MaxLine), MaxLine)
        for j in range(numLine):
            angleP = np.random.uniform(-MaxAngle, MaxAngle)
            angle = startAngle + angleP if j % 2 == 0 else startAngle + angleP + np.pi
            length = np.random.uniform(0.4 * MaxLength, MaxLength)
            thickness = np.random.randint(int(0.8 * MaxThin), MaxThin)
            # clamp to value of end point.
            endX = max(0, min(int(startX + length * np.sin(angle)), h - 1))
            endY = max(0, min(int(startY + length * np.cos(angle)), w - 1))
            # cut the stroke if the colors are different.
            cv2.line(stroke, (startX, startY), (endX, endY), 1.0, thickness=thickness)
            startX, startY = endX, endY
    stroke = np.expand_dims(stroke, -1)
    return stroke


"""
Unit-test code.
"""
# from PIL import Image
# from torchvision import transforms
# #
# transform_train = transforms.Compose([
#                 transforms.ToTensor(),
#             ])
# #
# img = np.asarray(Image.open('../Data/DAVIS/JPEGImages/480p/weightlifting/00001.jpg'), dtype=np.float32)
# sketch = np.asarray(Image.open('../Data/DAVIS/sketchs/weightlifting/00001.jpg'), dtype=np.float32)
# color = np.asarray(Image.open('../Data/DAVIS/colors/weightlifting/00001.jpg'), dtype=np.float32)
# mask = maskGenerator(img)
# stroke = strokeGenerator(img)
# color = color * stroke
#
# IMG = 2. * transform_train(img / 255) - 1.  # rescale the value of Img into (-1, 1).
# SKE = transform_train(sketch)
# COL = transform_train(color)
# MASK = transform_train(mask)