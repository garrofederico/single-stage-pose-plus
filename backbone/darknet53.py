"""
    DarkNet-53 for ImageNet-1K, implemented in PyTorch.
    Original source: 'YOLOv3: An Incremental Improvement,' https://arxiv.org/abs/1804.02767.
"""

__all__ = ['DarkNet53', 'darknet53']

import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .common import conv1x1_block, conv3x3_block
from .utils import sigmaBranch, fusionBranch, smooth_func, xdog_func, ordinaryConvs, SENet, XDogNet

import cv2
import numpy as np

class DarkUnit(nn.Module):
    """
    DarkNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    alpha : float
        Slope coefficient for Leaky ReLU activation.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 alpha):
        super(DarkUnit, self).__init__()
        assert (out_channels % 2 == 0)
        mid_channels = out_channels // 2

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            activation=nn.LeakyReLU(
                negative_slope=alpha,
                inplace=True))
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=nn.LeakyReLU(
                negative_slope=alpha,
                inplace=True))

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + identity


class DarkNet53(nn.Module):
    """
    DarkNet-53 model from 'YOLOv3: An Incremental Improvement,' https://arxiv.org/abs/1804.02767.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    alpha : float, default 0.1
        Slope coefficient for Leaky ReLU activation.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 alpha=0.1,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(DarkNet53, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            activation=nn.LeakyReLU(
                negative_slope=alpha,
                inplace=True)))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                if j == 0:
                    stage.add_module("unit{}".format(j + 1), conv3x3_block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=2,
                        activation=nn.LeakyReLU(
                            negative_slope=alpha,
                            inplace=True)))
                else:
                    stage.add_module("unit{}".format(j + 1), DarkUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        alpha=alpha))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
            # 
            #xdogs = XDogNet(out_channels)
            #self.features.add_module("xdogs{}".format(i + 1), xdogs)
            #
            #if i <= 3:
                #sigma = sigmaBranch(out_channels)
                #self.features.add_module("sigma{}".format(i + 1), sigma)
                #fusion = fusionBranch(out_channels)
                #self.features.add_module("fusion{}".format(i + 1), fusion)
                #
                #ordConvs = ordinaryConvs(out_channels)
                #self.features.add_module("ordConvs{}".format(i + 1), ordConvs)
                #
                #seLayer = SENet(out_channels)
                #self.features.add_module("seLayer{}".format(i + 1), seLayer)
                #
                #pass
            #
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
                    
    def forward(self, x):
        # x = self.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.output(x)
        # return x

        out0 = self.features.init_block(x)
        out1 = self.features.stage1(out0)
        out2 = self.features.stage2(out1)
        out3 = self.features.stage3(out2)
        out4 = self.features.stage4(out3)
        out5 = self.features.stage5(out4)
        
        # ordinaryConvs
        #if True:
        if False:
            out0 = self.features.init_block(x)
            out1 = self.features.stage1(out0)
            out1 = self.features.ordConvs1(out1)
            #
            out2 = self.features.stage2(out1)
            out2 = self.features.ordConvs2(out2)
            #
            out3 = self.features.stage3(out2)
            out3 = self.features.ordConvs3(out3)
            #
            out4 = self.features.stage4(out3)
            out4 = self.features.ordConvs4(out4)
            #
            out5 = self.features.stage5(out4)
        
        # SENet layers
        #if True:
        if False:
            out0 = self.features.init_block(x)
            out1 = self.features.stage1(out0)
            out1 = out1 * self.features.seLayer1(out1)
            #
            out2 = self.features.stage2(out1)
            out2 = out2 * self.features.seLayer2(out2)
            #
            out3 = self.features.stage3(out2)
            out3 = out3 * self.features.seLayer3(out3)
            #
            out4 = self.features.stage4(out3)
            out4 = out4 * self.features.seLayer4(out4)
            #
            out5 = self.features.stage5(out4)
            
        # XDogNet Layers
        #if True:
        if False:
            out0 = self.features.init_block(x)
            # 
            out1 = self.features.stage1(out0)
            par1 = self.features.xdogs1(out1)
            out1 = xdog_func(out1, par1)
            #
            out2 = self.features.stage2(out1)
            par2 = self.features.xdogs2(out2)
            out2 = xdog_func(out2, par2)
            #
            out3 = self.features.stage3(out2)
            par3 = self.features.xdogs3(out3)
            out3 = xdog_func(out3, par3)
            #
            out4 = self.features.stage4(out3)
            par4 = self.features.xdogs4(out4)
            out4 = xdog_func(out4, par4)
            #
            out5 = self.features.stage5(out4)
            par5 = self.features.xdogs5(out5)
            out5 = xdog_func(out5, par5)

        # ours
        #if True:
        if False:
            out0 = self.features.init_block(x)
            # 
            out1 = self.features.stage1(out0)
            ba, ch, fh, fw = out1.shape
            wts1 = self.features.fusion1(out1)
            param1 = self.features.sigma1(out1)
            edge1 = torch.sigmoid(out1) / (torch.sigmoid(smooth_func(out1, param1)) + 1e-5)
            out1 = torch.unsqueeze(wts1[:,:,0],2)*out1.view(ba,ch,-1) + torch.unsqueeze(wts1[:,:,1],2)*edge1.view(ba,ch,-1)
            out1 = out1.view(ba,ch, fh, fw)
            # 
            out2 = self.features.stage2(out1)
            ba, ch, fh, fw = out2.shape
            wts2 = self.features.fusion2(out2)
            param2 = self.features.sigma2(out2)
            edge2 = torch.sigmoid(out2) / (torch.sigmoid(smooth_func(out2, param2)) + 1e-5)
            out2 = torch.unsqueeze(wts2[:,:,0],2)*out2.view(ba,ch,-1) + torch.unsqueeze(wts2[:,:,1],2)*edge2.view(ba,ch,-1)
            out2 = out2.view(ba,ch, fh, fw)
            # 
            out3 = self.features.stage3(out2)
            ba, ch, fh, fw = out3.shape
            wts3 = self.features.fusion3(out3)
            param3 = self.features.sigma3(out3)
            edge3 = torch.sigmoid(out3) / (torch.sigmoid(smooth_func(out3, param3)) + 1e-5)
            out3 = torch.unsqueeze(wts3[:,:,0],2)*out3.view(ba,ch,-1) + torch.unsqueeze(wts3[:,:,1],2)*edge3.view(ba,ch,-1)
            out3 = out3.view(ba,ch, fh, fw)
            # 
            out4 = self.features.stage4(out3)
            ba, ch, fh, fw = out4.shape
            wts4 = self.features.fusion4(out4)
            param4 = self.features.sigma4(out4)
            edge4 = torch.sigmoid(out4) / (torch.sigmoid(smooth_func(out4, param4)) + 1e-5)
            out4 = torch.unsqueeze(wts4[:,:,0],2)*out4.view(ba,ch,-1) + torch.unsqueeze(wts4[:,:,1],2)*edge4.view(ba,ch,-1)
            out4 = out4.view(ba,ch, fh, fw)

            # 
            out5 = self.features.stage5(out4)
            #ba, ch, fh, fw = out5.shape
            #wts5 = self.features.fusion5(out5)
            #param5 = self.features.sigma5(out5)
            #edge5 = torch.sigmoid(out5) / (torch.sigmoid(smooth_func(out5, param5)) + 1e-5)
            #out5 = torch.unsqueeze(wts5[:,:,0],2)*out5.view(ba,ch,-1) + torch.unsqueeze(wts5[:,:,1],2)*edge5.view(ba,ch,-1)
            #out5 = out5.view(ba,ch, fh, fw)

        #
        # # tmpCvImg = x[0,:,:,:].to('cpu').numpy().transpose(1,2,0)
        # tmpCvImg = out1[:,0,:,:].to('cpu').numpy().transpose(1,2,0)
        # tmpCvImg = cv2.normalize(tmpCvImg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # cv2.imshow("tmp1", tmpCvImg)

        # tmpCvImg = edge1[:,0,:,:].to('cpu').numpy().transpose(1,2,0)
        # tmpCvImg = cv2.normalize(tmpCvImg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # cv2.imshow("tmp2", tmpCvImg)

        # tmpCvImg = out3[:,0,:,:].to('cpu').numpy().transpose(1,2,0)
        # tmpCvImg = cv2.normalize(tmpCvImg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # cv2.imshow("tmp3", tmpCvImg)

        # tmpCvImg = edge3[:,0,:,:].to('cpu').numpy().transpose(1,2,0)
        # tmpCvImg = cv2.normalize(tmpCvImg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # cv2.imshow("tmp4", tmpCvImg)
        # cv2.waitKey(0)

        # 
        return [out1, out2, out3, out4, out5]


def get_darknet53(model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".torch", "models"),
                  **kwargs):
    """
    Create DarkNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    init_block_channels = 32
    layers = [2, 3, 9, 9, 5]
    channels_per_layers = [64, 128, 256, 512, 1024]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = DarkNet53(
        channels=channels,
        init_block_channels=init_block_channels,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def darknet53(**kwargs):
    """
    DarkNet-53 'Reference' model from 'YOLOv3: An Incremental Improvement,' https://arxiv.org/abs/1804.02767.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_darknet53(model_name="darknet53", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    pretrained = False

    models = [
        darknet53,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != darknet53 or weight_count == 41609928)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
