from math import sqrt

import torch
from torch import nn


class ConvBNRelu(nn.Module):
    def __init__(self, inp, oup, kernel, stride, padding):
        super(ConvBNRelu, self).__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(inp, oup, kernel, stride, padding, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))

    def forward(self, x):
        return self.cbr(x)


# ------------------------------------------------------------------------------
#  Class of Inverted Residual block
# ------------------------------------------------------------------------------

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expansion, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, dilation=dilation, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, dilation=dilation, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2Auto(nn.Module):
    def __init__(self, in_channels, cfg=None, expansion_cfg=None, num_classes=1000):
        super(MobileNetV2Auto, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        if cfg is None:
            cfg = [32, 16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320, 1280]
        if expansion_cfg is None:
            expansion_cfg = [None, 1] + [6] * 16 + [None]

        interverted_residual_setting = [None,
                                        [None, None, 1, 1],
                                        [None, None, 1, 2],
                                        [None, None, 1, 1],
                                        [None, None, 1, 2],
                                        [None, None, 1, 1],
                                        [None, None, 1, 1],
                                        [None, None, 1, 2],
                                        [None, None, 1, 1],
                                        [None, None, 1, 1],
                                        [None, None, 1, 1],
                                        [None, None, 1, 1],
                                        [None, None, 1, 1],
                                        [None, None, 1, 1],
                                        [None, None, 1, 2],
                                        [None, None, 1, 1],
                                        [None, None, 1, 1],
                                        [None, None, 1, 1],
                                        None]

        # 根据cfg，配置interverted_residual_setting
        for i, v in enumerate(cfg):
            if i == 0 or i == len(cfg) - 1:  # in & out
                interverted_residual_setting[i] = v
            else:
                interverted_residual_setting[i][1] = v

        # 根据expansion_cfg，配置
        for i, v in enumerate(expansion_cfg):
            if i == 0 or i == len(cfg) - 1:
                continue
            interverted_residual_setting[i][0] = v
        # print(interverted_residual_setting)

        # 1. building first layer
        input_channel, last_channel = interverted_residual_setting[0], interverted_residual_setting[-1]
        self.last_channel = last_channel
        self.features = [ConvBNRelu(self.in_channels, input_channel, 3, 2, 1)]

        # 2. building inverted residual blocks
        for t, c, n, s in interverted_residual_setting[1:-1]:
            output_channel = c
            for i in range(n):  # block的重复次数为n
                if i == 0:  # 除第一个block外，每一个父block中的第一个子block进行降采样，即stride = 2
                    self.features.append(InvertedResidual(input_channel, output_channel, s, expansion=t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, expansion=t))
                input_channel = output_channel

        # 3.building last several layers
        self.features.append(ConvBNRelu(input_channel, self.last_channel, 1, 1, 0))

        # make it nn.Sequential
        # 将input feature、InvertedResidual、output feature三部分连接成sequential
        self.features = nn.Sequential(*self.features)

        if self.num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.last_channel, num_classes),
            )

        # Initialize weights
        self._init_weights()

    def forward(self, x):
        # Stage1
        x = self.features[0](x)
        x = self.features[1](x)
        # Stage2
        x = self.features[2](x)
        x = self.features[3](x)
        # Stage3
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        # Stage4
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)
        x = self.features[13](x)
        # Stage5
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)
        x = self.features[17](x)
        x = self.features[18](x)

        # Output
        return x

    def _load_pretrained_model(self, pretrained_file):
        pretrain_dict = torch.load(pretrained_file, map_location='cpu')
        model_dict = {}
        state_dict = self.state_dict()
        print("[MobileNetV2] Loading pretrained model...")
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
            else:
                print(k, "is ignored")
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
