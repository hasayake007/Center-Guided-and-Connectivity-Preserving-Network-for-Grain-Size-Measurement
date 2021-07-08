import skimage
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from soft_skeleton import soft_skel


class soft_cldice(nn.Module):
    def __init__(self, iter_= 3, smooth = 1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(y_true, y_pred):
        skel_pred = soft_skel(y_pred, iters)
        skel_true = soft_skel(y_true, iters)
        tprec = (torch.sum(torch.mul(skel_pred, y_true)[:,1:,:,:,:])+ smooth)/(torch.sum(skel_pred[:,1:,:,:,:])+smooth)
        tsens = (torch.sum(torch.mul(skel_true, y_pred)[:,1:,:,:,:])+ smooth)/(torch.sum(skel_true[:,1:,:,:,:])+smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]
    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]
    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred), dim=(2, 3))
    coeff = (2. * intersection + smooth) / (torch.sum(y_true, dim=(2, 3)) + torch.sum(y_pred, dim=(2, 3)) + smooth)
    return torch.mean(1. - coeff)



class soft_dice_cldice(nn.Module):
    def __init__(self, alpha=0.5, smooth = 1.):
        super(soft_dice_cldice, self).__init__()
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        # dice = soft_dice(y_true, y_pred)
        skel_pred = soft_skel(y_pred, 10)
        skel_true = soft_skel(y_true, 10)
        tprec = (torch.sum(torch.mul(skel_pred, y_true), dim=(2, 3))+self.smooth)/(torch.sum(skel_pred, dim=(2, 3)) + self.smooth)
        tsens = (torch.sum(torch.mul(skel_true, y_pred), dim=(2, 3))+self.smooth)/(torch.sum(skel_true, dim=(2, 3)) + self.smooth)
        cl_dice = 1. - 2.0*(tprec*tsens)/(tprec+tsens)
        # return torch.mean((1.0-self.alpha)*dice+self.alpha*cl_dice)
        return torch.mean(cl_dice)


class my_cldice(nn.Module):
    def __init__(self):
        super(my_cldice, self).__init__()

    def forward(self, y_true, y_pred, skel_true, skel_pred):
        y_pred = nn.functional.sigmoid(y_pred)
        tprec = (torch.sum(torch.mul(skel_pred, y_true), dim=(2, 3)) + 1.)/(torch.sum(skel_pred, dim=(2, 3)) + 1.)
        tsens = (torch.sum(torch.mul(skel_true, y_pred), dim=(2, 3)) + 1.)/(torch.sum(skel_true, dim=(2, 3)) + 1.)
        cl_dice = 1. - 2.0*(tprec*tsens)/(tprec+tsens)
        return torch.mean(cl_dice)


