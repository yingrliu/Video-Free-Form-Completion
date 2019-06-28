"""
Author: Yingru Liu
Implementation of basic operations, including gated convolution, gated transposed convolution
and spectral-normalized convolution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', lrn=True):
        """
        In this implementation, dilation must be a scalar.
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param padding_mode:
        :param lrn:
        """
        super(GatedConv2d, self).__init__()
        self.ConvA = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                               padding_mode)
        self.ConvB = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                             bias, padding_mode),
                                    nn.Sigmoid()
                                   )
        self.lrn = nn.LocalResponseNorm(size=5, alpha=1., beta=0.5) if lrn else None
        self.dilation, self.kernel_size = dilation, kernel_size
        return

    def forward(self, input):
        if self.dilation > 1:
            ph, pw = int(self.dilation * (self.kernel_size[0] - 1) / 2), \
                     int(self.dilation * (self.kernel_size[1] - 1) / 2)
            input = F.pad(input, [pw, pw, ph, ph])
        A = self.ConvA(input)
        B = self.ConvB(input)
        if self.lrn:
            A = self.lrn(A)
        return F.leaky_relu(A) * B


class GatedConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros'):
        """
        Unlike GatedConv2d, there is not LRN in this class.
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param output_padding:
        :param groups:
        :param bias:
        :param dilation:
        :param padding_mode:
        """
        super(GatedConvTranspose2d, self).__init__()
        self.ConvTransposeA = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups,
                 bias, dilation, padding_mode)
        self.ConvTransposeB = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                                      output_padding, groups, bias, dilation, padding_mode),
                                   nn.Sigmoid()
                                   )
        return

    def forward(self, input):
        A = self.ConvTransposeA(input)
        B = self.ConvTransposeB(input)
        return F.leaky_relu(A) * B


class SNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', Ip=1):
        super(SNConv2d, self).__init__()
        self.Ip = Ip
        self.Conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                             groups, bias, padding_mode), n_power_iterations=Ip)
        return

    def forward(self, input):
        return self.Conv(input)