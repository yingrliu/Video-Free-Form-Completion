"""
Author: Yingru Liu
This file contains the basic toolkit to generate sketch, mask and colormap of a given image.
"""

import cv2
import os
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torchvision import transforms
from Toolkit.deeplabv3.modeling.deeplabv3p import DeepLabV3p
#

def sketchGenerator(img):
    """
    use the Canny Edge detector to generate image sketch.
    :param img: RGB image.
    :return: sketch: the ndarray-form of the sketch.
    :return: sketch_tensor: the torch tensor form of the sketch. used for training or inference.
    """
    gs_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gs_img = cv2.equalizeHist(gs_img)
    mean = np.mean(gs_img)
    sketch = np.asarray(cv2.Canny(gs_img, 0.25*mean, 1.25*mean), dtype=np.float32)
    sketch_tensor = torch.from_numpy(sketch).unsqueeze(0)
    return sketch, sketch_tensor

#######################################################################################################################
# Define DeepLabV3p and load pre-trained parameters.
# model = DeepLabV3p(num_classes=13, output_stride=16, sync_bn=True, freeze_bn=False)
# if torch.cuda.is_available():
#     model = model.cuda()
# abs_path = os.path.dirname(os.path.abspath(__file__))
# if not os.path.isfile(os.path.join(abs_path, 'deeplabv3/checkpoints/checkpoint.pth.tar')):
#     raise RuntimeError("=> no checkpoint found at '{}'".format(
#         os.path.join(abs_path, 'deeplabv3/checkpoints/checkpoint.pth.tar')))
# if torch.cuda.is_available():
#     checkpoint = torch.load(os.path.join(abs_path, 'deeplabv3/checkpoints/checkpoint.pth.tar'))
# else:
#     checkpoint = torch.load(os.path.join(abs_path, 'deeplabv3/checkpoints/checkpoint.pth.tar'), map_location='cpu')
# state_dict = {}
# # The parameters saved in checkpoints have preix "module."
# for key, param in checkpoint['state_dict'].items():
#     if key[0:7] != 'module.':
#         continue
#     state_dict[key[7:]] = param
# model.load_state_dict(state_dict)
# model.eval()
# # transforms.ToTensor()
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     ]
# )

def colorGenerator_by_Parser(img):
    """
    use the pretrained DeepLab-v3+ to extract the median color of each part of the person.
    The pytorch implementation of DeepLab-v3 from [https://github.com/jfzhang95/pytorch-deeplab-xception.git]
    is used.
    :param img: RGB image.
    :return: color: the ndarray-form of the color.
    :return: color_tensor: the torch tensor form of the color. used for training or inference.
    """
    #
    # mean = np.mean(img)
    # img = img - mean
    # img = img * 1.5 + mean * 0.75
    img = np.asarray(img / 255, dtype=np.float32)
    #
    tensor_img = transform(img)
    if torch.cuda.is_available():
        tensor_img = tensor_img.cuda()
    tensor_img_up = F.interpolate(tensor_img.unsqueeze(0), size=(512, 512))
    prediction = model.forward(tensor_img_up).detach()
    prediction = F.interpolate(prediction, size=img.shape[-3:-1])
    seg_tensor = torch.argmax(prediction, dim=-3, keepdim=True).squeeze(0)
    # retrieve the mean color of each segmentation as the output color.
    numSeg = prediction.size()[-3]
    color_tensor = torch.zeros_like(tensor_img)
    for i in range(numSeg):
        seg_mask = seg_tensor.eq(i)
        if torch.sum(seg_mask) != 0:
            median_color = torch.median(tensor_img.masked_select(seg_mask).view(3, -1), dim=-1)[0]
            color_tensor += median_color.unsqueeze(-1).unsqueeze(-1) * seg_mask.type(dtype=torch.float32)
    color = color_tensor.cpu().numpy().transpose((1, 2, 0))
    return color, color_tensor.detach()

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
    # mean = np.mean(img)
    # img = img - mean
    # img = img * 1.5 + mean * 0.75
    pixels = np.reshape(img, newshape=(-1, 3))
    kmean = KMeans(n_clusters=12, random_state=0).fit(pixels)
    labels, centroid = kmean.labels_, kmean.cluster_centers_
    for i in range(12):
        idx = np.where(labels == i)[0]
        pixels[idx] = centroid[i]
        print()
    #
    img = np.reshape(pixels, newshape=img.shape)
    return img

#######################################################################################################################
MaxThin = 10
MaxDraw = 10
MaxLine = 20
MaxAngle = 0.15 * np.pi
MaxLength = 50
def maskGenerator(img):
    h, w = img.shape[0], img.shape[1]
    mask = np.ones((h, w), dtype=np.float32)
    numDraw = np.random.randint(1, MaxDraw)
    for _ in range(numDraw):
        startX = np.random.randint(0, h)
        startY = np.random.randint(0, w)
        startAngle = np.random.uniform(0, 2 * np.pi)
        numLine = np.random.randint(0, MaxLine)
        for j in range(numLine):
            angleP = np.random.uniform(-MaxAngle, MaxAngle)
            angle = startAngle + angleP if j % 2 == 0 else startAngle + angleP + np.pi
            length = np.random.uniform(MaxLength)
            thickness = np.random.randint(0.8 * MaxThin, MaxThin)
            endX = int(startX + length * np.sin(angle))
            endY = int(startY + length * np.cos(angle))
            cv2.line(mask, (startX, startY), (endX, endY), 0.0, thickness=thickness)
            startX, startY = endX, endY
    return mask, torch.from_numpy(np.expand_dims(mask, 0))

"""
Unit-test code.
"""
import matplotlib.pyplot as plt
img = cv2.imread('../Data/DAVIS/JPEGImages/480p/bmx-bumps/00001.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
f, axarr = plt.subplots(2, 2)
[axi.set_axis_off() for axi in axarr.ravel()]
axarr[0, 0].imshow(img)
axarr[0, 0].set_title("Image")
sketch, sketch_tensor = sketchGenerator(img)
axarr[0, 1].imshow(sketch)
axarr[0, 1].set_title("Sketch")
# color, color_tensor = colorGenerator_by_Parser(img)
color = colorGenerator_by_Filter(img)
axarr[1, 0].imshow(color)
axarr[1, 0].set_title("Color-Map")
mask, mask_tensor = maskGenerator(img)
axarr[1, 1].imshow(mask)
axarr[1, 1].set_title("Mask")
plt.show()