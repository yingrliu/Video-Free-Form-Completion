"""
Author: Yingru Liu
This file contains the basic toolkit to generate sketch, mask and colormap of a given image.
"""

import cv2
import os
import torch
import numpy as np
from Toolkit.pytorch_hed.run import edge_detector
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torchvision import transforms
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
    gs_img = cv2.medianBlur(gs_img, 5)
    #
    sketch = np.asarray(gs_img, dtype=np.float32)
    sketch_tensor = torch.from_numpy(sketch).unsqueeze(0)
    return sketch, sketch_tensor

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
    color_tensor = torch.from_numpy(img).unsqueeze(0)
    return color, color_tensor


#######################################################################################################################
MaxThin = 25
MaxDraw = 15
MaxLine = 50
MaxAngle = 0.25 * np.pi
MaxLength = 50
def maskGenerator(img):
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
    return mask, torch.from_numpy(np.expand_dims(mask, 0))

"""
Unit-test code.
"""
# import matplotlib.pyplot as plt
# img = cv2.imread('../Data/DAVIS/JPEGImages/480p/choreography/00001.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# f, axarr = plt.subplots(2, 2)
# [axi.set_axis_off() for axi in axarr.ravel()]
# axarr[0, 0].imshow(img)
# axarr[0, 0].set_title("Image")
# sketch, _ = sketchGenerator(img)
# axarr[0, 1].imshow(sketch)
# axarr[0, 1].set_title("Sketch")
# # color, color_tensor = colorGenerator_by_Parser(img)
# color, _ = colorGenerator_by_Filter(img)
# axarr[1, 0].imshow(color)
# axarr[1, 0].set_title("Color-Map")
# mask, _ = maskGenerator(img)
# axarr[1, 1].imshow(mask)
# axarr[1, 1].set_title("Mask")
# plt.show()