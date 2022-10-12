from time import time
import numpy as np
import torch
import torch.nn as nn

from nni.compression.pytorch.utils import count_flops_params

from src.models.backbones.mobilenetv2_auto import InvertedResidual, ConvBNRelu
from src.models.modnet_auto import MODNet_auto, Conv2dIBNormRelu, SEBlock

from pruner.block import InvertResBlock, CBR, CIBRelu, LBlock
from pruner.prune import compute_weights, get_nums_of_keep_channels


def get_model_block(model):
    """
    :param model: 抠图模型
    :return:模型三个分支lr、hr、f的配置信息，包括结构名、层数、输入输出通道
    """
    backbone_blocks = []
    lr_blocks = []
    hr_blocks = []
    f_blocks = []

    layer_count = 0
    for idx, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, InvertedResidual):
            backbone_blocks.append(InvertResBlock(name, list(module.state_dict().values())))
        elif isinstance(module, ConvBNRelu):
            backbone_blocks.append(CBR(name, list(module.state_dict().values())))
        elif isinstance(module, SEBlock):
            lr_blocks.append(LBlock(name, list(module.state_dict().values())))
        elif isinstance(module, Conv2dIBNormRelu):
            if layer_count < 3:
                lr_blocks.append(CIBRelu(name, list(module.state_dict().values())))
            elif 3 <= layer_count < 16:
                hr_blocks.append(CIBRelu(name, list(module.state_dict().values())))
            else:
                f_blocks.append(CIBRelu(name, list(module.state_dict().values())))
            layer_count += 1

    model_block = backbone_blocks + lr_blocks + hr_blocks + f_blocks
    return model_block


def get_pruning_cfg(blocks, ratio=0.5):
    """
    固定比例与自适应剪枝
    :param blocks: 模型每一个分支的通道信息
    :param ratio: 剪枝比例默认为0.5
    :return: 剪枝后保留的通道数
    """
    cfg = []
    hr_in_cfg = []
    f_in_cfg = []

    ans = 0
    for block in blocks:
        if isinstance(block, CBR):  # mobilenet 的 first&last channel
            cfg.append(compute_weights(block.weight))
        elif isinstance(block, InvertResBlock):
            if block.num_layer == 2:
                cfg.append(cfg[-1])  # 第一个block
                cfg.append(compute_weights(block.output2_weight))
            else:
                nums_keep = max(compute_weights(block.output1_weight),
                                compute_weights(block.output2_weight))  # 为了group conv即保持一致，取最大值
                cfg.extend([nums_keep] * 2)
                cfg.append(compute_weights(block.output3_weight))
        elif isinstance(block, LBlock):
            cfg.append(int(cfg[-1] / 4))
            cfg.append(cfg[-2])
        elif isinstance(block, CIBRelu):
            if 'lr16x' in block.layer_name:
                cfg.append(compute_weights(block.weight))
            elif ans < 23 and ('lr8x' or 'lr' in block.layer_name):
                cfg.append(block.output_channel)
            else:
                cfg.append(get_nums_of_keep_channels(ratio, block.output_channel))
                if 23 <= ans <= 35:
                    hr_in_cfg.append(block.input_channel)
                else:
                    f_in_cfg.append(block.input_channel)
        ans += 1
    return cfg


def model_pruning(model, cfg, cfg_in):
    """
    根据cfg进行剪枝，并获得mask
    :param model: 待剪枝的模型
    :param cfg: 网络层输出通道的数量
    :param cfg_in: 网络层输入通道的配置
    """
    cfg_mask = []
    cfg_mask_for_hr_f_input = [None] * 57

    layer_id = 0
    for m in modnet.modules():
        if isinstance(m, nn.Conv2d):
            out_channels, in_channels = m.weight.data.shape[0], m.weight.data.shape[1]
            if out_channels == cfg[layer_id]:
                cfg_mask.append(torch.ones(out_channels))
                if layer_id >= 57 and in_channels == cfg_in[layer_id]:
                    cfg_mask_for_hr_f_input.append(torch.ones(in_channels))

                if layer_id != 69 and layer_id != 73:  # 对hr last 特判，对input channel剪枝
                    layer_id += 1
                    continue

            # hr与f中均存在一边倒的情况
            if out_channels == cfg[layer_id] and (layer_id != 69 and layer_id != 73):
                cfg_mask.append(torch.ones(out_channels))
            if in_channels == cfg_in[layer_id]:
                cfg_mask_for_hr_f_input.append(torch.ones(in_channels))

            weight_copy = m.weight.data.abs().clone().cpu().numpy()
            if out_channels != cfg[layer_id]:
                L1_norm = np.sum(weight_copy, axis=(1, 2, 3))  # 基于l1范数，将除了第一个维度以外的三维度上的权重相加
                arg_max = np.argsort(L1_norm)  # 将该layer的所有filter升序排列
                arg_max_rev = arg_max[::-1][:cfg[layer_id]]  # 降序，并获取前cfg[layer_id]个filter
                mask = torch.zeros(out_channels)  # mask处理，初始化为0，表示去除的部分
                mask[arg_max_rev.tolist()] = 1  # arg_max_rev中的所有filter置1，保留
                cfg_mask.append(mask)  # 将这一layer的mask保存，layer id自增，处理下一层
            if layer_id >= 57 and in_channels != cfg_in[layer_id]:  # input channel pruning（针对hr与f）
                L1_norm = np.sum(weight_copy, axis=(0, 2, 3))
                arg_max = np.argsort(L1_norm)
                arg_max_rev = arg_max[::-1][:cfg_in[layer_id]]
                assert arg_max_rev.size == cfg_in[layer_id], "size of arg_max_rev not correct"
                mask = torch.zeros(in_channels)
                mask[arg_max_rev.tolist()] = 1
                cfg_mask_for_hr_f_input.append(mask)
            layer_id += 1
        elif isinstance(m, nn.Linear):  # 由于对mobilenet second Linear out 作了剪枝处理，因此需要制作对应的mask
            out_features = m.weight.data.shape[0]
            weight_copy = m.weight.data.abs().clone()
            weight_copy = weight_copy.cpu().numpy()
            L1_norm = np.sum(weight_copy, axis=1)
            arg_max = np.argsort(L1_norm)  # 将该layer的所有filter升序排列
            arg_max_rev = arg_max[::-1][:cfg[layer_id]]  # 降序，并获取前cfg[layer_id]个filter
            assert arg_max_rev.size == cfg[layer_id], "size of arg_max_rev not correct"  # 断言，检查剪枝后的features数量是否与设定的一致
            mask = torch.zeros(out_features)  # mask处理，初始化为0，表示去除的部分
            mask[arg_max_rev.tolist()] = 1  # arg_max_rev中的所有filter置1，保留
            cfg_mask.append(mask)  # 将这一layer的mask保存，layer id自增，处理下一层
            layer_id += 1

    return cfg_mask, cfg_mask_for_hr_f_input


# 8.替换weight与bias
def param_substitution(model, new_model, cfg_mask, cfg_mask_for_hr_f_input):
    start_mask = torch.ones(3)
    layer_id_in_cfg = 0
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(modnet.modules(), new_modnet.modules()):
        if isinstance(m0, nn.Conv2d):
            if m0.weight.data.shape == m1.weight.data.shape:  # 如果shape相同，不作prune，但需要对weight以及bias赋值，然后跳过该layer的处理
                # 特殊的，当遇到branch边界时，如lr_branch与hr_branch，需考虑mask的衔接性；
                # 由于lr_branch的last layer为conv，与先前的conv后 + BN以及Relu不一致，因此在BN层的mask替换便不存在；
                # lr_branch的输出为单通道，且hr_branch输入、输出尺寸分别为16,32；
                # 由于id自增在weight以及bias替换之后，所以在id为56时就需要替换（hr_branch对应id为57时）
                if layer_id_in_cfg == 56:  # flag1
                    layer_id_in_cfg += 1
                    start_mask = cfg_mask_for_hr_f_input[layer_id_in_cfg]
                    end_mask = cfg_mask[layer_id_in_cfg]
                m1.weight.data = m0.weight.data
                if m1.bias is not None:  # modify
                    m1.bias.data = m0.bias.data
                continue

            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))  # 获取input channel
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))  # 待保留的通道id，即为output channel
            print('Layer {:d}  [Conv2d]  In shape: {:d}, Out shape {:d}.'.format(layer_id_in_cfg, idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))

            if m0.groups == 1:  # 普通卷积
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()  # 获取原模型中的Conv input tensor
            else:
                w1 = m0.weight.data.clone()

            w1 = w1[idx1.tolist(), :, :, :].clone()  # 获取目标filter
            m1.weight.data = w1.clone()

            if m0.bias is not None:
                m1.bias.data = m0.bias.data[idx1.tolist()]

            if layer_id_in_cfg == 69:  # flag2:与flag1类似的情况
                layer_id_in_cfg += 1
                start_mask = cfg_mask_for_hr_f_input[layer_id_in_cfg]
                end_mask = cfg_mask[layer_id_in_cfg]

        elif isinstance(m0, nn.BatchNorm2d):
            if layer_id_in_cfg <= 53:  # id53前都为CBR结构
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                m1.num_batches_tracked = m0.num_batches_tracked.clone()  # add
                layer_id_in_cfg += 1
                start_mask = end_mask
                end_mask = cfg_mask[layer_id_in_cfg]
            else:  # 此时为CBI结构，折半处理（但需要考虑结构的特殊性，后期应修改结构），该BN于IBNorm相连，因此不对id赋值
                idx = int(np.argwhere(np.asarray(end_mask.cpu().numpy())).size / 2)
                m1.weight.data = m0.weight.data[:idx].clone()
                m1.bias.data = m0.bias.data[:idx].clone()
                m1.running_mean = m0.running_mean[:idx].clone()
                m1.running_var = m0.running_var[:idx].clone()
                m1.num_batches_tracked = m0.num_batches_tracked.clone()  # add
        elif isinstance(m0, nn.InstanceNorm2d):
            if m0.num_features != m1.num_features:
                idx = int(np.argwhere(np.asarray(end_mask.cpu().numpy())).size / 2)
                m1.num_features = idx  # 该层affine为False，不存在weight以及bias，因此直接对num_features赋值
            layer_id_in_cfg += 1

            start_mask = end_mask if layer_id_in_cfg < 57 else cfg_mask_for_hr_f_input[layer_id_in_cfg]
            end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Linear):  # 连续两个Linear
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))  # in_features
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))  # out_features
            print('Layer {:d}  [Linear]  In shape: {:d}, Out shape {:d}.'.format(layer_id_in_cfg, idx0.size, idx1.size))
            w1 = m0.weight.data[:, idx0.tolist()].clone()  # 根据上一层layer，获取该层的in_features
            w1 = w1[idx1.tolist(), :].clone()  # 获取out_features
            m1.weight.data = w1.clone()
            layer_id_in_cfg += 1
            start_mask = end_mask
            end_mask = cfg_mask[layer_id_in_cfg]

    return new_modnet


if __name__ == '__main__':
    # 1.加载模型
    modnet = MODNet_auto(backbone_pretrained=False)
    pretrained_ckpt = './pretrained/new_modnet_photographic_portrait_matting.ckpt'
    modnet.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(pretrained_ckpt).items()})

    # 2.获取block，并根据ratio自动获取保留通道数
    blocks = get_model_block(modnet)

    # 3.自适应剪枝、固定比例剪枝，获取剪枝后返回的通道数cfg
    ratio = 0.5
    cfg = get_pruning_cfg(blocks, ratio=ratio)

    # 4.1 根据cfg，进一步提取构建mobilenet v2所需的my_cfg
    my_cfg = []
    new_cfg = cfg[:-22]
    for i in range(2, len(new_cfg), 3):
        my_cfg.append(new_cfg[i])
    my_cfg = [new_cfg[0]] + my_cfg + [new_cfg[-1]]
    # print(my_cfg)

    # 4.2 进一步提取构建mobilenet v2 block中所需的expansion
    residual_cfg = new_cfg[2:-2]
    expansion_cfg = [1]
    for i in range(0, len(residual_cfg), 3):
        m = int(residual_cfg[i + 1] / residual_cfg[i])
        expansion_cfg.append(m)
        residual_cfg[i], residual_cfg[i + 1], residual_cfg[i + 2] = residual_cfg[i], int(m * residual_cfg[i]), int(
            m * residual_cfg[i])  # 更新cfg
    # print(expansion_cfg)

    # 4.3 可以进行一些优化，不再加上首尾的None
    my_expansion_cfg = [None] + expansion_cfg + [None]
    cfg = new_cfg[:2] + residual_cfg + new_cfg[-2:] + cfg[-22:]  # 将mobilenet所需的cfg与其余部分结合，得到modnet的cfg

    # 4.4 三个branch的input cfg
    my_lr_channels = [cfg[55], cfg[54]]
    my_hr_channels = [cfg[2], cfg[8]]
    my_f_channels = [cfg[55]]

    # 4.5 配置hr与f的输入，构建cfg_in
    hr_channels = int(32 * ratio)
    hr_in_cfg = [my_hr_channels[0], hr_channels + 3, my_hr_channels[1], 2 * hr_channels, 2 * hr_channels + 32 + 3,
                 2 * hr_channels, 2 * hr_channels, 2 * hr_channels, 2 * hr_channels, hr_channels, hr_channels,
                 hr_channels + 3, hr_channels]
    f_in_cfg = [my_f_channels[0], 2 * hr_channels, hr_channels + 3, int(hr_channels / 2)]
    cfg_in = [None] * 57 + hr_in_cfg + f_in_cfg

    # 5.构建模型。
    new_modnet = MODNet_auto(cfg=my_cfg, expansion=my_expansion_cfg, lr_channel=my_lr_channels,
                             hr_channel=my_hr_channels,
                             f_channel=my_f_channels,
                             hr_channels=int(32 * (1 - ratio)),
                             backbone_pretrained=False)

    # 利用NNI测试new_model的可用性
    # dummy_input = torch.randn([1, 3, 512, 512])
    # flops, params, _ = count_flops_params(new_modnet, dummy_input, verbose=True)
    # print(f"Model FLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M")

    # 6.剪枝，获取mask
    cfg_mask, cfg_mask_for_hr_f_input = model_pruning(modnet, cfg, cfg_in)

    # 7.将modnet参数更新至new_modnet（权值替换）
    new_modnet = param_substitution(modnet, new_modnet, cfg_mask, cfg_mask_for_hr_f_input)

    # torch.save(new_modnet.state_dict(), './result/pruned_model.ckpt')

    # 9.测试权值替换后的new_modnet
    dummy_input = torch.randn([1, 3, 512, 512])
    flops, params, _ = count_flops_params(new_modnet, dummy_input, verbose=False)
    print(f"\nPruned Model FLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M")

    # 10.推理速度比较
    t1 = time()
    res1 = modnet(dummy_input)
    delta1 = time() - t1
    print(f"\nInfer time of modnet: {delta1:.3f}s")

    t2 = time()
    res2 = new_modnet(dummy_input)
    delta2 = time() - t2
    print(f"Infer time of new_modnet: {delta2:.3f}s")
    print(f"Improvement of inference speed: {delta1 / delta2:.2f}-fold")
