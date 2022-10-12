"""
用于构建模型的网络块，如下：
1.CBR（Conv + BatchNorm + Relu）
2.CIBRelu（Conv + InstanceNorm+ Relu）
3.LBlock（Linear）
4.InvertResBlock
"""


class BasicBlock:
    def __init__(self, layer_name: str, state_dict: list):
        self.layer_name = layer_name
        self.state_dict = [s for s in state_dict if len(s.shape) != 0]
        self.input_channel = 0
        self.output_channel_one = 0
        self.output_channel = 0
        self.num_layer = 0
        self.weight = 0
        self.type = None

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "name={}, ".format(self.layer_name)
        s += "layer_num={},".format(
            self.num_layer)  # attention，first block only have two layer;if it is CBR, layer num is zero
        if self.num_layer == 2:
            s += "out_channel_one={},".format(self.output_channel_one)
        else:
            s += "in_channel={},".format(self.input_channel)
        s += "out_channel={})".format(self.output_channel)

        return s


class CBR(BasicBlock):
    def __init__(self, layer_name: str, state_dict: list):
        super().__init__(layer_name, state_dict)
        self.input_channel = self.state_dict[0].shape[1]
        self.output_channel = self.state_dict[-1].shape[0]
        self.weight = self.state_dict[0]


class LBlock(BasicBlock):
    def __init__(self, layer_name: str, state_dict: list):
        super().__init__(layer_name, state_dict)
        self.input_channel = self.state_dict[0].shape[1]  # modify
        self.output_channel = self.state_dict[0].shape[0]


# HR branch中，尽管如conv_hr2x是包含层级关系的，但每一层依旧可以被提取
class CIBRelu(BasicBlock):
    def __init__(self, layer_name: str, state_dict: list):
        super().__init__(layer_name, state_dict)
        self.input_channel = self.state_dict[0].shape[1]
        self.output_channel = self.state_dict[0].shape[0]
        self.weight = self.state_dict[0]


class InvertResBlock(BasicBlock):
    def __init__(self, layer_name: str, state_dict: list):
        super().__init__(layer_name, state_dict)
        self.input_channel = self.state_dict[0].shape[1]
        self.output_channel_one = self.state_dict[0].shape[0]
        self.output_channel = self.state_dict[-1].shape[0]
        self.num_layer = len(self.state_dict) // 5  # Conv2d只含有1个weight，BN含有weight/bias/running_mean等5个属性
        self.output1_weight = self.state_dict[0]
        self.output2_weight = self.state_dict[5]
        if self.num_layer != 2:
            self.output3_weight = self.state_dict[10]
