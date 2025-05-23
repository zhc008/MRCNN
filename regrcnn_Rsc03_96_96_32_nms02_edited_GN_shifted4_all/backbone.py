#!/usr/bin/env python
# Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch.nn as nn
import torch.nn.functional as F


class ConvGenerator():
    """conv-layer generator to avoid 2D vs. 3D distinction in code.
    """

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, c_in, c_out, ks, pad=0, stride=1, norm=None, relu='relu'):
        """provides generic conv-layer modules for set dimension.
        :param c_in: number of in_channels.
        :param c_out: number of out_channels.
        :param ks: kernel size.
        :param pad: pad size.
        :param stride: kernel stride.
        :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
        :param relu: string specifying type of nonlinearity. If None, no nonlinearity is applied.
        :return: 2D or 3D conv-layer module.
        """

        if self.dim == 2:
            module = nn.Conv2d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride)
            if norm is not None:
                if norm == 'instance_norm':
                    norm_layer = nn.InstanceNorm2d(c_out)
                elif norm == 'batch_norm':
                    norm_layer = nn.BatchNorm2d(c_out)
                elif norm == 'group_norm': # changes here, added group normalization
                    norm_layer = nn.GroupNorm(2, c_out)
                else:
                    raise ValueError('norm type as specified in configs is not implemented...')
                module = nn.Sequential(module, norm_layer)

        elif self.dim==3:
            module = nn.Conv3d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride)
            if norm is not None:
                if norm == 'instance_norm':
                    norm_layer = nn.InstanceNorm3d(c_out)
                elif norm == 'batch_norm':
                    norm_layer = nn.BatchNorm3d(c_out)
                elif norm == 'group_norm': # changes here, added group normalization
                    norm_layer = nn.GroupNorm(2, c_out) # normalize with 2 groups
                else:
                    raise ValueError('norm type as specified in configs is not implemented... {}'.format(norm))
                module = nn.Sequential(module, norm_layer)
        else:
            raise Exception("Invalid dimension {} in conv-layer generation.".format(self.dim))

        if relu is not None:
            if relu == 'relu':
                relu_layer = nn.ReLU(inplace=True)
            elif relu == 'leaky_relu':
                relu_layer = nn.LeakyReLU(inplace=True)
            else:
                raise ValueError('relu type as specified in configs is not implemented...')
            module = nn.Sequential(module, relu_layer)

        return module

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x

class ResBlock(nn.Module):

    def __init__(self, start_filts, planes, end_filts, conv, stride=1, identity_skip=True, norm=None, relu='relu'):
        """Builds a residual net block.
        :param start_filts: #input channels to the block.
        :param planes: #channels in block's hidden layers. set start_filts>planes<end_filts for bottlenecking.
        :param end_filts: #output channels of the block.
        :param conv: conv-layer generator.
        :param stride:
        :param identity_skip: whether to use weight-less identity on skip-connection if no rescaling necessary.
        :param norm:
        :param relu:
        """
        super(ResBlock, self).__init__()

        self.conv1 = conv(start_filts, planes, ks=1, stride=stride, norm=norm, relu=relu)
        self.conv2 = conv(planes, planes, ks=3, pad=1, norm=norm, relu=relu)
        self.conv3 = conv(planes, end_filts, ks=1, norm=norm, relu=None)
        if relu == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif relu == 'leaky_relu':
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            raise Exception("Chosen activation {} not implemented.".format(self.relu))

        if stride!=1 or start_filts!=end_filts or not identity_skip:
            self.scale_residual = conv(start_filts, end_filts, ks=1, stride=stride, norm=norm, relu=None)
        else:
            self.scale_residual = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.scale_residual:
            residual = self.scale_residual(x)
        else:
            residual = x
        out += residual
        out = self.relu(out)
        return out


class FPN(nn.Module):
    """
    Feature Pyramid Network from https://arxiv.org/pdf/1612.03144.pdf with options for modifications.
    by default is constructed with Pyramid levels P2, P3, P4, P5.
    """
    def __init__(self, cf, conv, relu_enc="relu", relu_dec=None, operate_stride1=False):
        """
        :param conv: instance of custom conv class containing the dimension info.
        :param relu_enc: string specifying type of nonlinearity in encoder. If None, no nonlinearity is applied.
	    :param relu_dec: same as relu_enc but for decoder.
        :param operate_stride1: boolean flag. enables adding of Pyramid levels P1 (output stride 2) and P0 (output stride 1).
        from configs:
	        :param channels: len(channels) is nr of channel dimensions in input data.
	        :param start_filts:  number of feature_maps in first layer. rest is scaled accordingly.
	        :param end_filts: number of feature_maps for output_layers of all levels in decoder.
	        :param res_architecture: string deciding whether to use "resnet50" or "resnet101".
	        :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
	        :param sixth_pooling: boolean flag. enables adding of Pyramid level P6.
        """
        super(FPN, self).__init__()

        self.start_filts, sf = cf.start_filts, cf.start_filts #sf = alias for readability
        self.out_channels = cf.end_filts
        self.n_blocks = [3, 4, {"resnet50": 6, "resnet101": 23}[cf.res_architecture], 3]
        self.block = ResBlock
        self.block_exp = 4 #factor by which to increase nr of channels in first block layer.
        self.relu_enc = relu_enc
        self.relu_dec = relu_dec
        self.operate_stride1 = operate_stride1
        self.sixth_pooling = cf.sixth_pooling


        if operate_stride1:
            self.C0 = nn.Sequential(conv(len(cf.channels), sf, ks=3, pad=1, norm=cf.norm, relu=relu_enc),
                                    conv(sf, sf, ks=3, pad=1, norm=cf.norm, relu=relu_enc))

            self.C1 = conv(sf, sf, ks=7, stride=(2, 2, 1) if conv.dim == 3 else 2, pad=3, norm=cf.norm,
                           relu=relu_enc)

        else:
            self.C1 = conv(len(cf.channels), sf, ks=7, stride=(2, 2, 1) if conv.dim == 3 else 2, pad=3, norm=cf.norm,
                           relu=relu_enc)

        C2_layers = []
        C2_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if
                         conv.dim == 2 else nn.MaxPool3d(kernel_size=3, stride=(2, 2, 1), padding=1))
        C2_layers.append(self.block(sf, sf, sf*self.block_exp, conv=conv, stride=1, norm=cf.norm,
                                    relu=relu_enc))
        for i in range(1, self.n_blocks[0]):
            C2_layers.append(self.block(sf*self.block_exp, sf, sf*self.block_exp, conv=conv,
                                        stride=1, norm=cf.norm, relu=relu_enc))
        self.C2 = nn.Sequential(*C2_layers)

        C3_layers = []
        C3_layers.append(self.block(sf*self.block_exp, sf*2, sf*self.block_exp*2, conv=conv,
                                    stride=2, norm=cf.norm, relu=relu_enc))
        for i in range(1, self.n_blocks[1]):
            C3_layers.append(self.block(sf*self.block_exp*2, sf*2, sf*self.block_exp*2,
                                        conv=conv, norm=cf.norm, relu=relu_enc))
        self.C3 = nn.Sequential(*C3_layers)

        C4_layers = []
        C4_layers.append(self.block(sf*self.block_exp*2, sf*4, sf*self.block_exp*4,
                                    conv=conv, stride=2, norm=cf.norm, relu=relu_enc))
        for i in range(1, self.n_blocks[2]):
            C4_layers.append(self.block(sf*self.block_exp*4, sf*4, sf*self.block_exp*4,
                                        conv=conv, norm=cf.norm, relu=relu_enc))
        self.C4 = nn.Sequential(*C4_layers)

        C5_layers = []
        C5_layers.append(self.block(sf*self.block_exp*4, sf*8, sf*self.block_exp*8,
                                    conv=conv, stride=2, norm=cf.norm, relu=relu_enc))
        for i in range(1, self.n_blocks[3]):
            C5_layers.append(self.block(sf*self.block_exp*8, sf*8, sf*self.block_exp*8,
                                        conv=conv, norm=cf.norm, relu=relu_enc))
        self.C5 = nn.Sequential(*C5_layers)

        if self.sixth_pooling:
            C6_layers = []
            C6_layers.append(self.block(sf*self.block_exp*8, sf*16, sf*self.block_exp*16,
                                        conv=conv, stride=2, norm=cf.norm, relu=relu_enc))
            for i in range(1, self.n_blocks[3]):
                C6_layers.append(self.block(sf*self.block_exp*16, sf*16, sf*self.block_exp*16,
                                            conv=conv, norm=cf.norm, relu=relu_enc))
            self.C6 = nn.Sequential(*C6_layers)

        if conv.dim == 2:
            self.P1_upsample = Interpolate(scale_factor=2, mode='bilinear')
            self.P2_upsample = Interpolate(scale_factor=2, mode='bilinear')
        else:
            self.P1_upsample = Interpolate(scale_factor=(2, 2, 1), mode='trilinear')
            self.P2_upsample = Interpolate(scale_factor=(2, 2, 1), mode='trilinear')

        if self.sixth_pooling:
            self.P6_conv1 = conv(sf*self.block_exp*16, self.out_channels, ks=1, stride=1, relu=relu_dec)
        self.P5_conv1 = conv(sf*self.block_exp*8, self.out_channels, ks=1, stride=1, relu=relu_dec)
        self.P4_conv1 = conv(sf*self.block_exp*4, self.out_channels, ks=1, stride=1, relu=relu_dec)
        self.P3_conv1 = conv(sf*self.block_exp*2, self.out_channels, ks=1, stride=1, relu=relu_dec)
        self.P2_conv1 = conv(sf*self.block_exp, self.out_channels, ks=1, stride=1, relu=relu_dec)
        self.P1_conv1 = conv(sf, self.out_channels, ks=1, stride=1, relu=relu_dec)

        if operate_stride1:
            self.P0_conv1 = conv(sf, self.out_channels, ks=1, stride=1, relu=relu_dec)
            self.P0_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=relu_dec)

        self.P1_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=relu_dec)
        self.P2_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=relu_dec)
        self.P3_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=relu_dec)
        self.P4_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=relu_dec)
        self.P5_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=relu_dec)

        if self.sixth_pooling:
            self.P6_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=relu_dec)


    def forward(self, x):
        """
        :param x: input image of shape (b, c, y, x, (z))
        :return: list of output feature maps per pyramid level, each with shape (b, c, y, x, (z)).
        """
        if self.operate_stride1:
            c0_out = self.C0(x)
        else:
            c0_out = x

        c1_out = self.C1(c0_out)
        c2_out = self.C2(c1_out)
        c3_out = self.C3(c2_out)
        c4_out = self.C4(c3_out)
        c5_out = self.C5(c4_out)
        if self.sixth_pooling:
            c6_out = self.C6(c5_out)
            p6_pre_out = self.P6_conv1(c6_out)
            p5_pre_out = self.P5_conv1(c5_out) + F.interpolate(p6_pre_out, scale_factor=2)
        else:
            p5_pre_out = self.P5_conv1(c5_out)
        #pre_out means last step before prediction output
        p4_pre_out = self.P4_conv1(c4_out) + F.interpolate(p5_pre_out, scale_factor=2)
        p3_pre_out = self.P3_conv1(c3_out) + F.interpolate(p4_pre_out, scale_factor=2)
        p2_pre_out = self.P2_conv1(c2_out) + F.interpolate(p3_pre_out, scale_factor=2)

        # plot feature map shapes for debugging.
        # for ii in [c0_out, c1_out, c2_out, c3_out, c4_out, c5_out, c6_out]:
        #     print ("encoder shapes:", ii.shape)
        #
        # for ii in [p6_out, p5_out, p4_out, p3_out, p2_out, p1_out]:
        #     print("decoder shapes:", ii.shape)

        p2_out = self.P2_conv2(p2_pre_out)
        p3_out = self.P3_conv2(p3_pre_out)
        p4_out = self.P4_conv2(p4_pre_out)
        p5_out = self.P5_conv2(p5_pre_out)
        out_list = [p2_out, p3_out, p4_out, p5_out]

        if self.sixth_pooling:
            p6_out = self.P6_conv2(p6_pre_out)
            out_list.append(p6_out)

        if self.operate_stride1:
            p1_pre_out = self.P1_conv1(c1_out) + self.P2_upsample(p2_pre_out)
            p0_pre_out = self.P0_conv1(c0_out) + self.P1_upsample(p1_pre_out)
            # p1_out = self.P1_conv2(p1_pre_out) # usually not needed.
            p0_out = self.P0_conv2(p0_pre_out)
            out_list = [p0_out] + out_list

        return out_list