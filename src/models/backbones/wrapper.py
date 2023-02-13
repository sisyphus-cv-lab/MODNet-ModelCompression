import os
from functools import reduce

import torch
import torch.nn as nn

from .mobilenetv2 import MobileNetV2
from .mobilenetv2_auto import MobileNetV2Auto


class BaseBackbone(nn.Module):
    def __init__(self, in_channels):
        super(BaseBackbone, self).__init__()
        self.in_channels = in_channels

        self.model = None
        self.enc_channels = []

    def forward(self, x):
        raise NotImplementedError

    def load_pretrained_ckpt(self):
        raise NotImplementedError


class BaseBackboneAuto(nn.Module):
    def __init__(self, in_channels, cfg, expansion_cfg):
        super(BaseBackboneAuto, self).__init__()
        self.in_channels = in_channels
        self.cfg = cfg
        self.expansion_cfg = expansion_cfg

        self.model = None
        self.enc_channels = []

    def forward(self, x):
        raise NotImplementedError

    def load_pretrained_ckpt(self):
        raise NotImplementedError


class MobileNetV2Backbone(BaseBackbone):
    """ MobileNetV2 Backbone
    """

    def __init__(self, in_channels):
        super(MobileNetV2Backbone, self).__init__(in_channels)

        self.model = MobileNetV2(self.in_channels, alpha=1.0, expansion=6, num_classes=None)
        self.enc_channels = [16, 24, 32, 96, 1280]

    def forward(self, x):
        # x = reduce(lambda x, n: self.model.features[n](x), list(range(0, 2)), x)
        x = self.model.features[0](x)
        x = self.model.features[1](x)
        enc2x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(2, 4)), x)
        x = self.model.features[2](x)
        x = self.model.features[3](x)
        enc4x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(4, 7)), x)
        x = self.model.features[4](x)
        x = self.model.features[5](x)
        x = self.model.features[6](x)
        enc8x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(7, 14)), x)
        x = self.model.features[7](x)
        x = self.model.features[8](x)
        x = self.model.features[9](x)
        x = self.model.features[10](x)
        x = self.model.features[11](x)
        x = self.model.features[12](x)
        x = self.model.features[13](x)
        enc16x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(14, 19)), x)
        x = self.model.features[14](x)
        x = self.model.features[15](x)
        x = self.model.features[16](x)
        x = self.model.features[17](x)
        x = self.model.features[18](x)
        enc32x = x
        return [enc2x, enc4x, enc8x, enc16x, enc32x]

    def load_pretrained_ckpt(self):
        # the pre-trained model is provided by https://github.com/thuyngch/Human-Segmentation-PyTorch
        ckpt_path = './pretrained/mobilenetv2_human_seg.ckpt'
        if not os.path.exists(ckpt_path):
            print('cannot find the pretrained mobilenetv2 backbone')
            exit()

        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt)


class MobileNetV2BackboneAuto(BaseBackboneAuto):
    """ MobileNetV2 Auto Backbone
    """

    def __init__(self, in_channels, cfg, expansion_cfg):
        super(MobileNetV2BackboneAuto, self).__init__(in_channels, cfg, expansion_cfg)

        self.model = MobileNetV2Auto(self.in_channels, self.cfg, self.expansion_cfg, num_classes=None)
        self.last_channel = self.model.last_channel

    def forward(self, x):
        x = self.model.features[0](x)
        x = self.model.features[1](x)
        enc2x = x

        x = self.model.features[2](x)
        x = self.model.features[3](x)
        enc4x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(4, 7)), x)
        x = self.model.features[4](x)
        x = self.model.features[5](x)
        x = self.model.features[6](x)
        enc8x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(7, 14)), x)
        x = self.model.features[7](x)
        x = self.model.features[8](x)
        x = self.model.features[9](x)
        x = self.model.features[10](x)
        x = self.model.features[11](x)
        x = self.model.features[12](x)
        x = self.model.features[13](x)
        enc16x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(14, 19)), x)
        x = self.model.features[14](x)
        x = self.model.features[15](x)
        x = self.model.features[16](x)
        x = self.model.features[17](x)
        x = self.model.features[18](x)

        #  3inverted + 1 inverted + CBR
        enc32x = x
        return [enc2x, enc4x, enc8x, enc16x, enc32x]

    def load_pretrained_ckpt(self):
        # the pre-trained model is provided by https://github.com/thuyngch/Human-Segmentation-PyTorch
        print('adw ')
        ckpt_path = './pretrained/new_mobilenetv2_human_seg.ckpt'
        if not os.path.exists(ckpt_path):
            print('cannot find the pretrained mobilenetv2 backbone')
            exit()

        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt)
