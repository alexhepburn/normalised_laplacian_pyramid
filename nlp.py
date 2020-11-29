"""
Normalised Laplacian Pyramid implemeneted in Pytorch
Author: Alex Hepburn <ah13558@bristol.ac.uk>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from gdn import GDN


LAPLACIAN_FILTER = np.array([[0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
                             [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                             [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
                             [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                             [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]],
                            dtype=np.float32)


class LaplacianPyramid(nn.Module):
    def __init__(self, k, dims=3, filt=None, trainable=False):
        super(LaplacianPyramid, self).__init__()
        if filt is None:
            filt = np.reshape(np.tile(LAPLACIAN_FILTER, (dims, 1, 1)),
                              (dims, 1, 5, 5))
        self.k = k
        self.trainable = trainable
        self.dims = dims
        self.filt = nn.Parameter(torch.Tensor(filt), requires_grad=False)
        self.dn_filts, self.sigmas = self.DN_filters()

    def DN_filters(self):
        sigmas = [0.0248, 0.0185, 0.0179, 0.0191, 0.0220, 0.2782]
        dn_filts = []
        dn_filts.append(torch.Tensor(np.reshape([[0, 0.1011, 0],
                                                 [0.1493, 0, 0.1460],
                                                 [0, 0.1015, 0.]]*self.dims,
                                                (self.dims,  1, 3, 3)).astype(np.float32)))

        dn_filts.append(torch.Tensor(np.reshape([[0, 0.0757, 0],
                                                 [0.1986, 0, 0.1846],
                                                 [0, 0.0837, 0]]*self.dims,
                                                (self.dims, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(torch.Tensor(np.reshape([[0, 0.0477, 0],
                                                 [0.2138, 0, 0.2243],
                                                 [0, 0.0467, 0]]*self.dims,
                                                (self.dims, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(torch.Tensor(np.reshape([[0, 0, 0],
                                                 [0.2503, 0, 0.2616],
                                                 [0, 0, 0]]*self.dims,
                                                (self.dims, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(torch.Tensor(np.reshape([[0, 0, 0],
                                                 [0.2598, 0, 0.2552],
                                                 [0, 0, 0]]*self.dims,
                                                (self.dims, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(torch.Tensor(np.reshape([[0, 0, 0],
                                                 [0.2215, 0, 0.0717],
                                                 [0, 0, 0]]*self.dims,
                                                (self.dims, 1, 3, 3)).astype(np.float32)))
        dn_filts = nn.ParameterList([nn.Parameter(x, requires_grad=self.trainable)
                                     for x in dn_filts])
        sigmas = nn.ParameterList([nn.Parameter(torch.Tensor(np.array(x)),
                                                requires_grad=self.trainable) for x in sigmas])
        return dn_filts, sigmas

    def pyramid(self, im):
        out = []
        J = im
        pyr = []
        for i in range(0, self.k):
            J_padding_amount = conv_utils.pad([J.size(2), J.size(3)],
                                              self.filt.size(3), stride=2)
            I = F.conv2d(F.pad(J, J_padding_amount, mode='reflect'), self.filt,
                         stride=2, padding=0, groups=self.dims)
            I_up = F.interpolate(I, size=[J.size(2), J.size(3)],
                                 align_corners=True, mode='bilinear')
            I_padding_amount = conv_utils.pad([I_up.size(2), I_up.size(3)],
                                              self.filt.size(3), stride=1)
            I_up_conv = F.conv2d(F.pad(I_up, I_padding_amount, mode='reflect'),
                                 self.filt, stride=1, padding=0,
                                 groups=self.dims)
            out = J - I_up_conv
            out_padding_amount = conv_utils.pad(
                [out.size(2), out.size(3)], self.dn_filts[i].size(2), stride=1)
            out_conv = F.conv2d(
                F.pad(torch.abs(out), out_padding_amount, mode='reflect'),
                self.dn_filts[i],
                stride=1,
                groups=self.dims)
            out_norm = out / (self.sigmas[i]+out_conv)
            pyr.append(out_norm)
            J = I
        return pyr

    def compare(self, x1, x2):
        y1 = self.pyramid(x1)
        y2 = self.pyramid(x2)
        total = []
        # Calculate difference in perceptual space (Tensors are stored
        # strangley to avoid needing to pad tensors)
        for z1, z2 in zip(y1, y2):
            diff = (z1 - z2) ** 2
            sqrt = torch.sqrt(torch.mean(diff, (1, 2, 3)))
            total.append(sqrt)
        return torch.norm(torch.stack(total), 0.6)


class LaplacianPyramidGDN(nn.Module):
    def __init__(self, k, dims=3, filt=None):
        super(LaplacianPyramidGDN, self).__init__()
        if filt is None:
            filt = np.tile(LAPLACIAN_FILTER, (dims, 1, 1))
            filt = np.reshape(np.tile(LAPLACIAN_FILTER, (dims, 1, 1)),
                              (dims, 1, 5, 5))
        self.k = k
        self.dims = dims
        self.filt = nn.Parameter(torch.Tensor(filt))
        self.filt.requires_grad = False
        self.gdns = nn.ModuleList([expert_divisive_normalisation.GDN(
            dims, apply_independently=True) for i in range(self.k)])
        self.pad_one = nn.ReflectionPad2d(1)
        self.pad_two = nn.ReflectionPad2d(2)
        self.mse = nn.MSELoss(reduction='none')
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)

    def pyramid(self, im):
        J = im
        pyr = []
        for i in range(0, self.k):
            I = F.conv2d(self.pad_two(J), self.filt, stride=2, padding=0,
                         groups=self.dims)
            I_up = self.upsample(I)
            I_up_conv = F.conv2d(self.pad_two(I_up), self.filt, stride=1,
                                 padding=0, groups=self.dims)
            if J.size() != I_up_conv.size():
                I_up_conv = torch.nn.functional.interpolate(
                    I_up_conv, [J.size(2), J.size(3)])
            pyr.append(self.gdns[i](J - I_up_conv))
            J = I
        return pyr

    def compare(self, x1, x2):
        y1 = self.pyramid(x1)
        y2 = self.pyramid(x2)
        total = []
        # Calculate difference in perceptual space (Tensors are stored
        # strangley to avoid needing to pad tensors)
        for z1, z2 in zip(y1, y2):
            diff = (z1 - z2) ** 2
            sqrt = torch.sqrt(torch.mean(diff, (1, 2, 3)))
            total.append(sqrt)
        return torch.norm(torch.stack(total), 0.6)
