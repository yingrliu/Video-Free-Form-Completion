"""
Author: Yingru Liu
Implementation of SC-FEGAN in the paper:
SC-FEGAN: Face Editing Generative Adversarial Network with User's Sketch and Color
https://arxiv.org/abs/1902.06838
"""
import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
import numpy as np
from ImgModels.Ops.Layers import GatedConv2d, GatedConvTranspose2d, SNConv2d
from ImgModels.Ops.Trainer import _Trainer
from Toolkit.LossSet import *
from tqdm import tqdm
from torch.nn.modules.loss import _WeightedLoss

# The generator of sc_fegan.
class sc_fegan(nn.Module):
    def __init__(self, nfg = 64):
        super(sc_fegan, self).__init__()
        # Encoder.
        self.Conv1 = GatedConv2d(9, nfg, (7, 7), stride=2, padding=3, lrn=False)
        self.Conv2 = GatedConv2d(nfg, nfg*2, (5, 5), stride=2, padding=2)
        self.Conv3 = GatedConv2d(nfg*2, nfg*4, (5, 5), stride=2, padding=2)
        self.Conv4 = GatedConv2d(nfg*4, nfg*8, (3, 3), stride=2, padding=1)
        self.Conv5 = GatedConv2d(nfg * 8, nfg * 8, (3, 3), stride=2, padding=1)
        self.Conv6 = GatedConv2d(nfg * 8, nfg * 8, (3, 3), stride=2, padding=1)
        self.Conv7 = nn.Sequential(
            GatedConv2d(nfg * 8, nfg * 8, (3, 3), stride=2, padding=1),
            GatedConv2d(nfg * 8, nfg * 8, (3, 3), dilation=2),
            GatedConv2d(nfg * 8, nfg * 8, (3, 3), dilation=4),
            GatedConv2d(nfg * 8, nfg * 8, (3, 3), dilation=8),
            GatedConv2d(nfg * 8, nfg * 8, (3, 3), dilation=16),
        )
        # Decoder.
        self.Conv81 = GatedConvTranspose2d(nfg * 8, nfg * 8, (2, 2), stride=2)
        self.Conv82 = GatedConv2d(nfg*16, nfg*8, (3, 3), padding=1)
        #
        self.Conv91 = GatedConvTranspose2d(nfg * 8, nfg * 8, (2, 2), stride=2)
        self.Conv92 = GatedConv2d(nfg * 16, nfg * 8, (3, 3), padding=1)
        #
        self.Conv101 = GatedConvTranspose2d(nfg * 8, nfg * 8, (2, 2), stride=2)
        self.Conv102 = GatedConv2d(nfg * 16, nfg * 8, (3, 3), padding=1)
        #
        self.Conv111 = GatedConvTranspose2d(nfg * 8, nfg * 4, (2, 2), stride=2)
        self.Conv112 = GatedConv2d(nfg * 8, nfg * 4, (3, 3), padding=1)
        #
        self.Conv121 = GatedConvTranspose2d(nfg * 4, nfg * 2, (2, 2), stride=2)
        self.Conv122 = GatedConv2d(nfg * 4, nfg * 2, (3, 3), padding=1)
        #
        self.Conv131 = GatedConvTranspose2d(nfg * 2, nfg * 1, (2, 2), stride=2)
        self.Conv132 = GatedConv2d(nfg * 2, nfg * 1, (3, 3), padding=1)
        #
        self.Conv141 = GatedConvTranspose2d(nfg * 1, 3, (2, 2), stride=2)
        self.Conv142 = GatedConv2d(3 + 9, 3, (3, 3), lrn=False, padding=1)
        return

    def forward(self, input):
        # Encoding
        Conv1 = self.Conv1(input)
        Conv2 = self.Conv2(Conv1)
        Conv3 = self.Conv3(Conv2)
        Conv4 = self.Conv4(Conv3)
        Conv5 = self.Conv5(Conv4)
        Conv6 = self.Conv6(Conv5)
        Encode = self.Conv7(Conv6)      # size = 4 x 4.
        # Decoding
        Conv8 = self.Conv81(Encode)
        Conv8 = torch.cat([Conv8, Conv6], dim=-3)
        Conv8 = self.Conv82(Conv8)
        #
        Conv9 = self.Conv91(Conv8)
        Conv9 = torch.cat([Conv9, Conv5], dim=-3)
        Conv9 = self.Conv92(Conv9)
        #
        Conv10 = self.Conv101(Conv9)
        Conv10 = torch.cat([Conv10, Conv4], dim=-3)
        Conv10 = self.Conv102(Conv10)
        #
        Conv11 = self.Conv111(Conv10)
        Conv11 = torch.cat([Conv11, Conv3], dim=-3)
        Conv11 = self.Conv112(Conv11)
        #
        Conv12 = self.Conv121(Conv11)
        Conv12 = torch.cat([Conv12, Conv2], dim=-3)
        Conv12 = self.Conv122(Conv12)
        #
        Conv13 = self.Conv131(Conv12)
        Conv13 = torch.cat([Conv13, Conv1], dim=-3)
        Conv13 = self.Conv132(Conv13)
        #
        Conv14 = self.Conv141(Conv13)
        Conv14 = torch.cat([Conv14, input], dim=-3)
        Conv14 = self.Conv142(Conv14)
        return F.tanh(Conv14)


# the discriminator of sn_patch_gan
class sn_patch_gan(nn.Module):
    def __init__(self, nfg = 64):
        super(sn_patch_gan, self).__init__()
        self.net = nn.Sequential(
            SNConv2d(8, nfg, (5, 5), padding=2),                # layer 1.
            nn.LeakyReLU(),
            SNConv2d(nfg, nfg*2, (5, 5), stride=2, padding=2),      # layer 2.
            nn.LeakyReLU(),
            SNConv2d(nfg*2, nfg*4, (5, 5), stride=2, padding=2),      # layer 3.
            nn.LeakyReLU(),
            SNConv2d(nfg*4, nfg*4, (5, 5), stride=2, padding=2),      # layer 4.
            nn.LeakyReLU(),
            SNConv2d(nfg*4, nfg*4, (5, 5), stride=2, padding=2),      # layer 5.
            nn.LeakyReLU(),
            SNConv2d(nfg*4, nfg*4, (5, 5), stride=2, padding=2),      # layer 6.
            nn.Tanh()                          # In my opinion, if output is not in (-1, 1), the loss looks strange.
        )
        return

    def forward(self, input):
        return self.net(input)

class SC_FEGAN_Trainer(_Trainer):
    def __init__(self, args):
        # load default setting for unset arguments.
        args = _checkargs(args)
        #
        args.netG = sc_fegan()
        args.netD = sn_patch_gan()
        #
        super(SC_FEGAN_Trainer, self).__init__(args)
        # rewrite the loss function.
        self.lossG = _lossG(args.lambda_tv, args.lambda_per, args.lambda_sty, args.lambda_sn, args.lambda_gt)
        self.lossD = _lossD(args.lambda_hinge)
        #
        return

    def _train_epoch(self, epoch):
        tbar = tqdm(self.trainSet)
        ##########################
        loss_per_epoch, loss_sty_epoch, loss_tv_epoch, loss_mpixel_epoch, loss_sn_epoch, loss_gt_epoch = \
            [], [], [], [], [], []
        lossG_epoch, lossD_epoch = [], []
        ##########################
        for i, item in enumerate(tbar):
            Imgs, sketch, color, mask = item['Img'], item['Sketch'], item['Color'], item['Mask']
            sketch, color = (1. - mask) * sketch, (1. - mask) * color
            noise = (1. - mask) * torch.randn(size=mask.size())
            #
            if torch.cuda.is_available():
                Imgs, sketch, color, mask, noise = Imgs.cuda(), sketch.cuda(), color.cuda(), mask.cuda(), noise.cuda()
            #
            inputG = torch.cat([Imgs*mask, sketch, color, mask, noise], dim=-3)
            fake_Imgs = self.netG(inputG)
            comp_Imgs = Imgs * mask + fake_Imgs * (1. - mask)
            # update discriminator.
            lossD = []
            for _ in range(self.args.steps_dis):
                pred_comp = self.netD(torch.cat([comp_Imgs.detach(), sketch, color, mask], dim=-3))
                pred_gt = self.netD(torch.cat([Imgs, sketch, color, mask], dim=-3))
                lossD_gt, lossD_comp = self.lossD(pred_comp, pred_gt)
                lossD_ = lossD_gt + lossD_comp
                self.OptimD.zero_grad()
                lossD_gt.backward()
                lossD_comp.backward()
                self.OptimD.step()
                lossD.append(lossD_.detach().cpu().numpy())
            lossD = np.mean(lossD)
            # update generator.
            pred_comp = self.netD(torch.cat([comp_Imgs, sketch, color, mask], dim=-3))
            pred_gt = self.netD(torch.cat([Imgs, sketch, color, mask], dim=-3))
            loss_tv, loss_per, loss_sty, loss_mpixel, loss_sn, loss_gt = \
                self.lossG(fake_Imgs, comp_Imgs, Imgs, mask, pred_comp, pred_gt)
            lossG = loss_per + loss_sty + loss_tv + loss_mpixel + loss_sn + loss_gt
            self.OptimG.zero_grad()
            lossG.backward()
            self.OptimG.step()
            lossG = lossG.detach().cpu().numpy()
            # set the printing.
            tbar.set_description('Generator loss: {:6.4f}; Discriminator loss: {:6.4f}'.format(lossG, lossD))
            # save the records for the whole epoch.
            loss_per_epoch.append(loss_per.detach().cpu().numpy())
            loss_sty_epoch.append(loss_sty.detach().cpu().numpy())
            loss_tv_epoch.append(loss_tv.detach().cpu().numpy())
            loss_mpixel_epoch.append(loss_mpixel.detach().cpu().numpy())
            loss_sn_epoch.append(loss_sn.detach().cpu().numpy())
            loss_gt_epoch.append(loss_gt.detach().cpu().numpy())
            #
            lossG_epoch.append(lossG)
            lossD_epoch.append(lossD)
        # return the scalar records for visualization.
        return {'train/percetual': np.mean(loss_per_epoch),
                'train/style': np.mean(loss_sty_epoch),
                'train/total_variation': np.mean(loss_tv_epoch),
                'train/mask_per_pixel': np.mean(loss_mpixel_epoch),
                'train/G_SN': np.mean(loss_sn_epoch),
                'train/Ground_true_Square': np.mean(loss_gt_epoch),
                'train/lossG': np.mean(lossG_epoch),
                'train/lossD': np.mean(lossD_epoch),
                }

    def _evaluate_epoch(self, visualize=False):
        tbar = tqdm(self.testSet)
        vis_imgs = []
        for i, item in enumerate(tbar):
            Imgs, sketch, color, mask = item['Img'], item['Sketch'], item['Color'], item['Mask']
            sketch, color = (1. - mask) * sketch, (1. - mask) * color
            noise = (1. - mask) * torch.randn(size=mask.size())
            #
            if torch.cuda.is_available():
                Imgs, sketch, color, mask, noise = Imgs.cuda(), sketch.cuda(), color.cuda(), mask.cuda(), noise.cuda()
            #
            if visualize and len(vis_imgs) < 8 * 5:
                idx = 8 - len(vis_imgs) // 5
                # save the images.
                inputG = torch.cat([Imgs[-idx:] * mask[-idx:], sketch[-idx:], color[-idx:], mask[-idx:],
                                    noise[-idx:]], dim=-3).detach()
                fake_Imgs = self.netG(inputG).detach()
                comp_Imgs = Imgs[-idx:] * mask[-idx:] + fake_Imgs * (1. - mask[-idx:])
                for i in range(1, idx+1):
                    vis_imgs += [Imgs[-i]*mask[-i], mask[-i].repeat(3, 1, 1), sketch[-i].repeat(3, 1, 1),
                                 color[-i], comp_Imgs[-i]]
        return None, vis_imgs




def _checkargs(args):
    """
    check the item of the argumenets.
    :param args:
    :return:
    """
    # load defaul setting to some arguments.
    ################################################################################################################
    # load default setting for generator loss functions.
    if not hasattr(args, 'lambda_per'):
        warnings.warn("No lambda_per is specified for the Generator loss. Use default value (1e1).")
        args.lambda_per = 1e1
    if not hasattr(args, 'lambda_sty'):
        warnings.warn("No lambda_sty is specified for the Generator loss. Use default value (1e4).")
        args.lambda_sty = 1e4
    if not hasattr(args, 'lambda_tv'):
        warnings.warn("No lambda_tv is specified for the Generator loss. Use default value (10).")
        args.lambda_tv = 1e1
    if not hasattr(args, 'lambda_sn'):
        warnings.warn("No lambda_sn is specified for the Generator loss. Use default value (10).")
        args.lambda_sn = 1e1
    if not hasattr(args, 'lambda_gt'):
        warnings.warn("No lambda_gt is specified for the Generator loss. Use default value (10).")
        args.lambda_gt = 1e1
    ################################################################################################################
    # load default setting for discriminator loss functions.
    if not hasattr(args, 'lambda_hinge'):
        warnings.warn("No lambda_hinge is specified for the Discriminator loss. Use default value (1.).")
        args.lambda_hinge = 1.
    return args

class _lossG(nn.Module):
    def __init__(self, lambda_tv, lambda_per, lambda_sty, lambda_sn, lambda_gt, alpha=1.5):
        super(_lossG, self).__init__()
        self.loss_per_sty = [lambda_per, lambda_sty, perceptual_style_Loss()]
        self.loss_tv = [lambda_tv, TVLoss()]
        self.loss_mpixel = MaskedPixelLoss(alpha=alpha)
        self.loss_sn = [lambda_sn, SNLoss_G()]
        self.loss_gt = [lambda_gt, GtLoss()]
        return

    def forward(self, fake, comp, gt, mask, pred_comp, pred_gt):
        loss_per_fake_gt, loss_sty_fake_gt = self.loss_per_sty[-1](fake, gt)
        loss_per_comp_gt, loss_sty_comp_gt = self.loss_per_sty[-1](comp, gt)
        return self.loss_tv[0] * self.loss_tv[-1](comp), \
               self.loss_per_sty[0] * (loss_per_fake_gt + loss_per_comp_gt),\
               self.loss_per_sty[1] * (loss_sty_fake_gt + loss_sty_comp_gt), \
               self.loss_mpixel(fake, gt, mask),\
               self.loss_sn[0] * self.loss_sn[-1](pred_comp), \
               self.loss_gt[0] * self.loss_gt[-1](pred_gt)

class _lossD(nn.Module):
    def __init__(self, lambda_hinge):
        super(_lossD, self).__init__()
        self.loss_hinge = [lambda_hinge, SNLoss_D()]
        return

    def forward(self, comp, gt):
        term1, term2 = self.loss_hinge[-1](comp, gt)
        return self.loss_hinge[0] * term1, self.loss_hinge[0] *term2