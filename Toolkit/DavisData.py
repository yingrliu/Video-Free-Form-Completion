"""
Author: Yingru Liu
This file contains:
 1. an image dataset class that loads each frame as image and generate mask, sketch and color.
 2. an video dataset class that loads a sequence of images as video and generate only the mask, sketch and color for
    the first frame.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from Toolkit.PStoolkit import sketchGenerator, maskGenerator
from Toolkit.PStoolkit import colorGenerator_by_Filter as colorGenerator
from multiprocessing import Pool

abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ImgDir = os.path.join(abs_path, "data", "DAVIS", "JPEGImages", "480p")
MaskDir = os.path.join(abs_path, "data", "qd_imd")
ColImgDir = os.path.join(abs_path, "data", "DAVIS", "colors")
SketchDir = os.path.join(abs_path, "data", "DAVIS", "sketchs")
# split dataset.
train_list_path_1 = os.path.join(abs_path, "data", "DAVIS", "ImageSets", "2017", "train.txt")
train_list_path_2 = os.path.join(abs_path, "data", "DAVIS", "ImageSets", "2017", "val.txt")
train_list_path_3 = os.path.join(abs_path, "data", "DAVIS", "ImageSets", "2017", "test-dev.txt")
train_list_path_4 = os.path.join(abs_path, "data", "DAVIS", "ImageSets", "2019", "test-dev.txt")
train_list = [train_list_path_1, train_list_path_2, train_list_path_3, train_list_path_4]
#
test_list_path_1 = os.path.join(abs_path, "data", "DAVIS", "ImageSets", "2019", "test-challenge.txt")
test_list_path_2 = os.path.join(abs_path, "data", "DAVIS", "ImageSets", "2017", "test-challenge.txt")
test_list = [test_list_path_1, test_list_path_1]
#
mask_list_path_1 = os.path.join(abs_path, "data", "qd_imd", "test")
mask_list_path_2 = os.path.join(abs_path, "data", "qd_imd", "train")
mask_list = [mask_list_path_1, mask_list_path_2]


class ImgData(Dataset):
    def __init__(self, resize=None, train=True):
        listset = train_list if train else test_list
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
        self.transform = transform_train
        # build color dir and sketch dir.
        if not os.path.exists(ColImgDir):
            os.makedirs(ColImgDir)
        if not os.path.exists(SketchDir):
            os.makedirs(SketchDir)
        # acess the list of masks.
        for mask_folder in mask_list:
            files = os.listdir(mask_folder)
            for file in tqdm(files):
                file_path = os.path.join(mask_folder, file)
                assert os.path.isfile(file_path) and file_path[-3:] == 'png'
                self.masks.append(file_path)
        # access the list of images.
        for imagelist in listset:
            with open(imagelist, 'r') as f:
                classes = f.read().splitlines()
                for cla in classes:
                    #
                    ImgSubDir = os.path.join(ImgDir, cla)
                    ColSubDir = os.path.join(ColImgDir, cla)
                    SketchSubDir = os.path.join(SketchDir, cla)
                    if not os.path.exists(ColSubDir):
                        os.makedirs(ColSubDir)
                    if not os.path.exists(SketchSubDir):
                        os.makedirs(SketchSubDir)
                    #
                    files = os.listdir(ImgSubDir)
                    for file in files:
                        file_path = os.path.join(ImgSubDir, file)
                        assert os.path.isfile(file_path) and file_path[-3:] == 'jpg'
                        print(file_path)
                        self.files.append(file_path)
                        # access color.
                        color_path = os.path.join(ColSubDir, file)
                        self.colors.append(color_path)
                        if not os.path.exists(color_path):
                            img = cv2.imread(file_path)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            color = colorGenerator(img)[0]
                            cv2.imwrite(color_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
                        # access sketch.
                        sketch_path = os.path.join(SketchSubDir, file)
                        self.sketches.append(sketch_path)
                        if not os.path.exists(sketch_path):
                            img = cv2.imread(file_path)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            sketch = sketchGenerator(img)[0]
                            cv2.imwrite(sketch_path, sketch)
        return

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        Img = Image.open(self.files[idx])
        sketch_tensor = Image.open(self.sketches[idx])
        color_tensor = Image.open(self.colors[idx])
        rand_idx = np.random.randint(low=0, high=len(self.masks))
        mask_tensor = Image.open(self.masks[rand_idx])
        return {'Img': 2. * self.transform(Img) - 1.,  # rescale the value of Img into (-1, 1).
                'Sketch': self.transform(sketch_tensor),
                'Color': self.transform(color_tensor),
                'Mask': self.transform(mask_tensor)
                }

# multi-processing.
def get_color_sketch(cla):
    #
    ImgSubDir = os.path.join(ImgDir, cla)
    ColSubDir = os.path.join(ColImgDir, cla)
    SketchSubDir = os.path.join(SketchDir, cla)
    if not os.path.exists(ColSubDir):
        os.makedirs(ColSubDir)
    if not os.path.exists(SketchSubDir):
        os.makedirs(SketchSubDir)
    files = os.listdir(ImgSubDir)
    for file in files:
        file_path = os.path.join(ImgSubDir, file)
        assert os.path.isfile(file_path) and file_path[-3:] == 'jpg'
        # access color.
        color_path = os.path.join(ColSubDir, file)
        if not os.path.exists(color_path):
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            color = colorGenerator(img)[0]
            cv2.imwrite(color_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        # access sketch.
        sketch_path = os.path.join(SketchSubDir, file)
        if not os.path.exists(sketch_path):
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sketch = sketchGenerator(img)[0]
            cv2.imwrite(sketch_path, sketch)
    return

def preprocessing(listset):
    """
    preprocess the datasets.
    :param listset: the list of process file.
    :return:
    """
    # access the list of images.
    for imagelist in listset:
        with open(imagelist, 'r') as f:
            classes = f.read().splitlines()
            pool = Pool(processes=4)
            pool.map(get_color_sketch, classes)
    return



def mp_preprocessing():
    preprocessing(train_list)
    preprocessing(test_list)
