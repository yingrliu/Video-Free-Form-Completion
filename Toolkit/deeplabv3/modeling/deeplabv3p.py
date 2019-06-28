import torch.nn as nn
import torch.nn.functional as F
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .aspp import build_aspp
from .decoder import build_decoder
from .backbone import build_backbone

class DeepLabV3p(nn.Module):
    def __init__(self,output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLabV3p, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(output_stride, BatchNorm)
        self.aspp = build_aspp(output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, BatchNorm)

        if freeze_bn:
            self.freeze_bn()
            self.freeze_bn()
    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

