from . import resnet

def build_backbone(output_stride, BatchNorm):
    return resnet.ResNet101(output_stride, BatchNorm)
