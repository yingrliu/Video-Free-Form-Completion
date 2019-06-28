"""
Author: Yingru Liu
This file contains the several loss definition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from copy import deepcopy
from torch.nn.modules.loss import _WeightedLoss

# To define perceptual loss and style loss we need to use the pre-trained VGG-16. However,
# the input of pre-trained model should be scaled to [0, 1] and normalize by mean and std.
# Therefore, the following layer is requried to add before the VGG-16.
class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        if torch.cuda.is_available():
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
        return

    def forward(self, img):
        # normalize img
        img = (img + 1.0) / 2
        return (img - self.mean) / self.std

class perceptual_style_Loss(nn.Module):
    """
    The perceptual and style loss for SC-FEGAN.
    """
    def __init__(self, per_layers=('pool1', 'pool2', 'pool3'),
                 sty_layers=('pool1', 'pool2', 'pool3'), norm=1):
        """

        :param weight:
        :param size_average:
        :param ignore_index:
        :param reduce:
        :param reduction:
        :param layers: the layers of VGG-16 to compute the loss.
        """
        super(perceptual_style_Loss, self).__init__()
        self.normal = Normalization()
        if torch.cuda.is_available():
            self.normal = self.normal.cuda()
        self.vgg16 = models.vgg16_bn(pretrained=True)
        if torch.cuda.is_available():
            self.vgg16 = self.vgg16.cuda()
        self.vgg16.eval()
        table = {'pool1': 6, 'pool2': 13, 'pool3': 23, 'pool4': 33, 'poo5':43}
        self.per_layers = []
        self.sty_layers = []
        for layer in per_layers:
            assert layer in table.keys()
            self.per_layers.append(table[layer])
        for layer in sty_layers:
            assert layer in table.keys()
            self.sty_layers.append(table[layer])
        #
        self.per_layers = set(self.per_layers)
        self.sty_layers = set(self.sty_layers)
        self.lpLoss = nn.L1Loss() if norm == 1 else nn.MSELoss()
        return

    def _gram(self, feature):
        a, b, c, d = feature.size()
        features = feature.view(a * b, c * d)
        G = torch.mm(features, features.t())  # compute the gram product
        return G / (a * b * c * d)

    def forward(self, input1, input2):
        """
        input2 is the ground-true image.
        :param input1:
        :param input2:
        :return:
        """
        normal = deepcopy(self.normal)
        vgg16 = deepcopy(self.vgg16)
        if len(self.sty_layers)==0 and len(self.per_layers)==0:
            if torch.cuda.is_available():
                return torch.tensor(0).cuda(), torch.tensor(0).cuda()
            else:
                return torch.tensor(0), torch.tensor(0)
        per_loss, sty_loss = 0, 0
        feature1, feature2 = normal(input1), normal(input2).detach()
        for i in range(len(vgg16.features)):
            # stop the unnecessary computation.
            if i > max(max(self.per_layers), max(self.sty_layers)):
                break
            #
            feature1, feature2 = vgg16.features[i](feature1), vgg16.features[i](feature2).detach()
            if i in self.per_layers:
                per_loss += self.lpLoss(feature1, feature2)
            if i in self.sty_layers:
                sty_loss += self.lpLoss(self._gram(feature1), self._gram(feature2).detach())
        return per_loss, sty_loss



class TVLoss(nn.Module):
    """
    The total variation loss of SC-FEGAN.
    """
    def __init__(self, norm=1):
        super(TVLoss, self).__init__()
        self.norm = norm
        return

    def forward(self, input):
        col = input[:, :, 0:-1, :] - input[0, :, 1:, :]
        row = input[:, :, :, 0:-1] - input[:, :, :, 1:]
        col = torch.abs(col) if self.norm == 1 else col**2
        row = torch.abs(row) if self.norm == 1 else row**2
        return torch.mean(col) + torch.mean(row)

class MaskedPixelLoss(nn.Module):
    """
    The masked pixel loss (defined as per_pixel loss in the paper) of SC-FEGAN.
    """
    def __init__(self, alpha=1.5, norm=1):
        super(MaskedPixelLoss, self).__init__()
        self.lpLoss = nn.L1Loss() if norm == 1 else nn.MSELoss()
        self.alpha = alpha
        return

    def forward(self, input1, input2, mask):
        term1 = self.lpLoss(input1 * mask, input2 * mask)
        term2 = self.lpLoss(input1 * (1 - mask), input2 * (1 - mask))
        return term1 + self.alpha * term2

class SNLoss_G(_WeightedLoss):
    """
    The hinge loss of SN-PatchGAN for the generator.
    """
    def __init__(self):
        super(SNLoss_G, self).__init__()
        return

    def forward(self, input):
        """
        :param input: the prediction of the discriminator.
        :return:
        """
        return -torch.mean(input)


class GtLoss(_WeightedLoss):
    """
    The loss of SN-PatchGAN for the generator using the prediction of ground-true images.
    """
    def __init__(self):
        super(GtLoss, self).__init__()
        return

    def forward(self, input):
        """
        :param input: the prediction of the discriminator.
        :return:
        """
        return torch.mean(input**2)

class SNLoss_D(_WeightedLoss):
    """
    The hinge loss of SN-PatchGAN for the discriminator.
    """
    def __init__(self, use_relu=False):
        super(SNLoss_D, self).__init__()
        self.use_relu = use_relu
        return

    def forward(self, input1, input2):
        """
        :param input1:
        :param input2: prediction for ground-true image.
        :return:
        """
        term1 = torch.mean(1.0 - input2) if not self.use_relu else torch.mean(F.relu(1.0 - input2))
        term2 = torch.mean(1.0 + input1) if not self.use_relu else torch.mean(F.relu(1.0 + input1))
        return term1, term2