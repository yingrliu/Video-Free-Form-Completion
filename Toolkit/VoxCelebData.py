"""
Author: Yingru Liu
This file contains:
 1. an image dataset class that loads each frame as image and generate mask, sketch and color.
 2. an video dataset class that loads a sequence of images as video and generate only the mask, sketch and color for
    the first frame.
"""
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from Toolkit.PStoolkit import sketchGenerator, maskGenerator
from Toolkit.PStoolkit import colorGenerator_by_Parser as colorGenerator

abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ImgDir = os.path.join(abs_path, "data", "VoxCeleb1", "images")
MaskDir = os.path.join(abs_path, "data", "VoxCeleb1", "masks")
ColImgDir = os.path.join(abs_path, "data", "VoxCeleb1", "colors")
SketchDir = os.path.join(abs_path, "data", "VoxCeleb1", "sketchs")
if not os.path.exists(MaskDir):
    os.mkdir(MaskDir)
if not os.path.exists(ColImgDir):
    os.mkdir(ColImgDir)
if not os.path.exists(SketchDir):
    os.mkdir(SketchDir)
#

class ImgData(Dataset):
    def __init__(self, root_dir=ImgDir, resize=None):
        #
        if resize:
            transform_train = transforms.Compose([
                transforms.Resize(size=resize),
                transforms.ToTensor(),
            ]
            )
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ]
            )
        #
        self.files = []
        self.colors = []
        self.sketches = []
        self.masks = []
        self.resize = resize
        # access the list of images.
        for root, dirs, files in os.walk(root_dir, topdown=True):
            for i, file in enumerate(files):
                if i > 0:
                    break
                file_path = os.path.join(root, file)
                if file_path[-3:] == 'jpg':
                    assert os.path.exists(file_path)
                    print(file_path)
                    self.files.append(file_path)
                    subdir = os.path.split(root)
                    id = subdir[-1]
                    celeb = os.path.split(os.path.split(subdir[0])[0])[-1]
                    # generate color.
                    if not os.path.exists(os.path.join(ColImgDir, celeb, id)):
                        os.makedirs(os.path.join(ColImgDir, celeb, id))
                    color_path = os.path.join(ColImgDir, celeb, id, file)
                    if not os.path.exists(color_path):
                        img = cv2.imread(file_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        if isinstance(self.resize, tuple) or isinstance(self.resize, list) or \
                                isinstance(self.resize, np.ndarray):
                            img = cv2.resize(img, self.resize)
                        color = colorGenerator(img)[0]
                        plt.imsave(color_path, color)
                    self.colors.append(color_path)
                    # generate sketch.
                    if not os.path.exists(os.path.join(SketchDir, celeb, id)):
                        os.makedirs(os.path.join(SketchDir, celeb, id))
                    sketch_path = os.path.join(SketchDir, celeb, id, file)
                    if not os.path.exists(sketch_path):
                        img = cv2.imread(file_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        if isinstance(self.resize, tuple) or isinstance(self.resize, list) or \
                                isinstance(self.resize, np.ndarray):
                            img = cv2.resize(img, self.resize)
                        sketch = sketchGenerator(img)[0]
                        cv2.imwrite(sketch_path, sketch)
                    self.sketches.append(sketch_path)
                    # generate mask.
                    if len(self.masks) > 10000:
                        continue
                    mask_path = os.path.join(MaskDir, file)
                    if not os.path.exists(mask_path):
                        img = cv2.imread(file_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        if isinstance(self.resize, tuple) or isinstance(self.resize, list) or \
                                isinstance(self.resize, np.ndarray):
                            img = cv2.resize(img, self.resize)
                        mask = maskGenerator(img)[0]
                        cv2.imwrite(mask_path, 255 * mask)
                        #plt.imsave(mask_path, 1 - mask, cmap=cm.binary)
                    self.masks.append(mask_path)
        self.transform = transform_train
        return

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        Img = Image.open(self.files[idx])
        sketch_tensor = Image.open(self.sketches[idx])
        color_tensor = Image.open(self.colors[idx])
        rand_idx = np.random.randint(low=0, high=len(self.masks))
        mask_tensor = Image.open(self.masks[rand_idx])
        return {'Img': 2. * self.transform(Img) - 1.,         # rescale the value of Img into (-1, 1).
                'Sketch': self.transform(sketch_tensor),
                'Color': self.transform(color_tensor),
                'Mask': self.transform(mask_tensor)
                }



# todo:
class VideoData:
    def __init__(self, root_dir, transform, resize=None):
        self.files = []
        self.resize = resize
        # access the list of images.
        for root, dirs, files in os.walk(root_dir, topdown=True):
            if files and files[0][-3:] == 'jpg':
                print(root)
                self.files.append(root)
        self.transform = transform
        return

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        root = self.files[idx]
        jpg_files = os.listdir(root)
        Video = []
        for i, file in enumerate(jpg_files):
            Img = cv2.imread(os.path.join(root, file))
            Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
            if isinstance(self.resize, tuple) or isinstance(self.resize, list) or isinstance(self.resize, np.ndarray):
                Img = cv2.resize(Img, self.resize)
            if i == 0:
                sketch_tensor = sketchGenerator(Img)[1]
                color_tensor = colorGenerator(Img)[1]
                mask_tensor = maskGenerator(Img)[1]
            # transform Img into tensor.
            Img = 2. * self.transform(Img) - 1.
            Video.append(Img.unsqueeze(0))
        Video = torch.cat(Video, dim=0)
        # cause when initialize the class, the path in self.files contains at least 1 frame. Therefore,
        # sketch_tensor, color_tensor, mask_tensor always exist.
        return {'Video': Video,  # rescale the value of Img into (-1, 1).
                'Sketch': sketch_tensor,
                'Color': color_tensor,
                'Mask': mask_tensor
                }
