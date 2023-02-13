# Import the necessary packages
import os
import argparse
import numpy as np

import torch
import torch.nn as nn

from nni.compression.pytorch.utils import count_flops_params

from src.models.backbones.mobilenetv2_auto import InvertedResidual, ConvBNRelu
from src.models.modnet_auto import MODNet_auto, Conv2dIBNormRelu, SEBlock

from pruner.block import InvertResBlock, CBR, CIBRelu, LBlock
from pruner.prune import compute_weights, get_nums_of_keep_channels

# Defining constants
nums_of_backbone = 52
nums_of_cbr = 53
interval_of_residual = 3
nums_of_lr_branch = 57
last_layer_of_hr_branch = 69
last_layer_of_f_branch = 73
original_channels_of_hr_branch = 32

fus_layer_feature1 = 2
fus_layer_feature3 = 8
fus_layer_feature17 = 50
fus_layer_lr16x = 54
fus_layer_lr8x = 55


def get_model_block(model):
    """
    @param param model: model to be pruned
    :return:Configuration information of the three branches lr, hr, f of the model,
            including parameters, number of layers, input and output channels.
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
            # Add the CIBR to each of the three branches in MODNet
            if layer_count < 3:
                lr_blocks.append(CIBRelu(name, list(module.state_dict().values())))
            elif 3 <= layer_count < 16:
                hr_blocks.append(CIBRelu(name, list(module.state_dict().values())))
            else:
                f_blocks.append(CIBRelu(name, list(module.state_dict().values())))
            layer_count += 1

    model_block = backbone_blocks + lr_blocks + hr_blocks + f_blocks
    return model_block


def get_pruning_cfg(blocks, ratio, threshold):
    """
    Get the number of reserved channels by adaptive and fixed ratio pruning.
    @param blocks: channel information for each branch of the model
    @param ratio: pruning scale, for the filter of hr branch and f branch in MODNet
    @param threshold: pruning threshold for the adaptive part, for the MobileNetv2 part in the MODNet
    @return: number of output channels retained in each layer
    """
    model_out_cfg = []
    hr_in_cfg = []
    f_in_cfg = []

    cnt = 0
    for block in blocks:
        # first&last channel of backbone
        if isinstance(block, CBR):
            model_out_cfg.append(compute_weights(block.weight, threshold))
        elif isinstance(block, InvertResBlock):
            if block.num_layer == 2:
                model_out_cfg.append(model_out_cfg[-1])  # first block
                model_out_cfg.append(compute_weights(block.output2_weight, threshold))
            else:
                # The maximum value is taken, in order to ensure group conv
                nums_keep = max(compute_weights(block.output1_weight, threshold),
                                compute_weights(block.output2_weight, threshold))
                model_out_cfg.extend([nums_keep] * 2)
                model_out_cfg.append(compute_weights(block.output3_weight, threshold))
        elif isinstance(block, LBlock):
            model_out_cfg.append(int(model_out_cfg[-1] / 4))
            model_out_cfg.append(model_out_cfg[-2])
        elif isinstance(block, CIBRelu):
            if 'lr16x' in block.layer_name:
                model_out_cfg.append(compute_weights(block.weight, threshold))
            elif cnt < 23 and ('lr8x' or 'lr' in block.layer_name):
                model_out_cfg.append(block.output_channel)
            else:
                model_out_cfg.append(get_nums_of_keep_channels(ratio, block.output_channel))
                if 23 <= cnt <= 35:
                    hr_in_cfg.append(block.input_channel)
                else:
                    f_in_cfg.append(block.input_channel)
        cnt += 1
    return model_out_cfg


def model_pruning(model, cfg_out, cfg_in):
    """
    Get the mask according to the cfg
    @param model: model to be pruned
    @param cfg_out: The number of output channels of the network layer
    @param cfg_in: The number of input channels of the network layer
    @return: mask for model pruning, where 1 represents retention and 0 represents removal.
    """
    cfg_mask = []
    cfg_mask_of_hr_f_input = [torch.zeros(0)] * nums_of_lr_branch

    layer_id = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            out_channels, in_channels = m.weight.data.shape[0], m.weight.data.shape[1]
            if out_channels == cfg_out[layer_id]:
                cfg_mask.append(torch.ones(out_channels))
                if layer_id >= nums_of_lr_branch and in_channels == cfg_in[layer_id]:
                    cfg_mask_of_hr_f_input.append(torch.ones(in_channels))

                if layer_id != last_layer_of_hr_branch and layer_id != last_layer_of_f_branch:  # 对hr last 特判，对input channel剪枝
                    layer_id += 1
                    continue

            # a lopsided in hr and f branch
            if out_channels == cfg_out[layer_id] and (
                    layer_id != last_layer_of_hr_branch and layer_id != last_layer_of_f_branch):
                cfg_mask.append(torch.ones(out_channels))
            if in_channels == cfg_in[layer_id]:
                cfg_mask_of_hr_f_input.append(torch.ones(in_channels))

            weight_copy = m.weight.data.abs().clone().cpu().numpy()
            if out_channels != cfg_out[layer_id]:
                L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
                arg_max = np.argsort(L1_norm)
                arg_max_rev = arg_max[::-1][:cfg_out[layer_id]]
                mask = torch.zeros(out_channels)
                mask[arg_max_rev.tolist()] = 1
                cfg_mask.append(mask)

            # Considering the fusion of model, input channels for hr and f branches are processed
            if layer_id >= nums_of_lr_branch and in_channels != cfg_in[layer_id]:
                L1_norm = np.sum(weight_copy, axis=(0, 2, 3))
                arg_max = np.argsort(L1_norm)
                arg_max_rev = arg_max[::-1][:cfg_in[layer_id]]
                assert arg_max_rev.size == cfg_in[layer_id], "size of arg_max_rev not correct"
                mask = torch.zeros(in_channels)
                mask[arg_max_rev.tolist()] = 1
                cfg_mask_of_hr_f_input.append(mask)
            layer_id += 1

        # Processing of neurons in the fully connected connected to the convolutional layer
        elif isinstance(m, nn.Linear):
            out_features = m.weight.data.shape[0]
            weight_copy = m.weight.data.abs().clone()
            weight_copy = weight_copy.cpu().numpy()
            L1_norm = np.sum(weight_copy, axis=1)
            arg_max = np.argsort(L1_norm)
            arg_max_rev = arg_max[::-1][:cfg_out[layer_id]]
            assert arg_max_rev.size == cfg_out[
                layer_id], "size of arg_max_rev not correct"
            mask = torch.zeros(out_features)
            mask[arg_max_rev.tolist()] = 1
            cfg_mask.append(mask)
            layer_id += 1

    return cfg_mask, cfg_mask_of_hr_f_input


def param_substitution(model, new_model, cfg_mask, cfg_mask_of_hr_f_input, verbose=True):
    start_mask = torch.ones(3)
    layer_id_in_cfg = 0
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if isinstance(m0, nn.Conv2d):
            if m0.weight.data.shape == m1.weight.data.shape:
                if layer_id_in_cfg == nums_of_lr_branch - 1:
                    layer_id_in_cfg += 1
                    start_mask = cfg_mask_of_hr_f_input[layer_id_in_cfg]
                    end_mask = cfg_mask[layer_id_in_cfg]
                m1.weight.data = m0.weight.data
                if m1.bias is not None:
                    m1.bias.data = m0.bias.data
                continue

            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))

            if verbose:
                print(f'Layer {layer_id_in_cfg:d}  [Conv2d]  In shape: {idx0.size:d}, Out shape {idx1.size:d}.')

            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))

            # Regular convolution, get input tensor from original model
            if m0.groups == 1:
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            else:
                w1 = m0.weight.data.clone()

            # Removal of affected output feature maps
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()

            if m0.bias is not None:
                m1.bias.data = m0.bias.data[idx1.tolist()]

            # Consider mask cohesion
            if layer_id_in_cfg == last_layer_of_hr_branch:
                layer_id_in_cfg += 1
                start_mask = cfg_mask_of_hr_f_input[layer_id_in_cfg]
                end_mask = cfg_mask[layer_id_in_cfg]

        elif isinstance(m0, nn.BatchNorm2d):
            if layer_id_in_cfg <= nums_of_cbr:
                # Conv + BatchNorm + Relu
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
            else:
                # Conv + InstanceNorm + Relu
                idx = int(np.argwhere(np.asarray(end_mask.cpu().numpy())).size / 2)
                m1.weight.data = m0.weight.data[:idx].clone()
                m1.bias.data = m0.bias.data[:idx].clone()
                m1.running_mean = m0.running_mean[:idx].clone()
                m1.running_var = m0.running_var[:idx].clone()
                m1.num_batches_tracked = m0.num_batches_tracked.clone()  # add
        elif isinstance(m0, nn.InstanceNorm2d):
            if m0.num_features != m1.num_features:
                idx = int(np.argwhere(np.asarray(end_mask.cpu().numpy())).size / 2)
                m1.num_features = idx
            layer_id_in_cfg += 1

            start_mask = end_mask if layer_id_in_cfg < nums_of_lr_branch else cfg_mask_of_hr_f_input[layer_id_in_cfg]
            end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))

            if verbose:
                print('Layer {:d}  [Linear]  In shape: {:d}, Out shape {:d}.'.format(layer_id_in_cfg, idx0.size,
                                                                                     idx1.size))

            w1 = m0.weight.data[:, idx0.tolist()].clone()
            w1 = w1[idx1.tolist(), :].clone()
            m1.weight.data = w1.clone()
            layer_id_in_cfg += 1
            start_mask = end_mask
            end_mask = cfg_mask[layer_id_in_cfg]

    return new_model


def run(pretrained_path, ratio, threshold, weight_replace=False, verbose=True, save_dir='./result/'):
    """
    Pruning pipeline
    @param pretrained_path: model weights to be pruned
    @param ratio: pruning scale, for the filter of hr branch and f branch in MODNet
    @param threshold: pruning threshold for the adaptive part, for the MobileNetv2 part in the MODNet
    @param weight_replace: depends on whether to fine-tune or train from scratch after pruning
    @param verbose: used to print or not the channel changes for each layer
    @param save_dir: the directory where the model is saved after pruning

    We only publish the parameter replacement when ratio and threshold are within a certain range,
    other cases are recommended not to replace.

    The range is as follows:
    1、When ratio is 0.5, threshold >= 0.1 && threshold <= 0.5
    2、The best parameter is: ratio=0.5,threshold=0.5.
    """
    condition1 = (ratio == 0.5 and threshold > 0.5 and weight_replace)
    condition2 = (ratio != 0.5 and weight_replace)
    if ratio == 0.5:
        assert not condition1, f"Weights replacing, expected threshold to be between (0, 0.6), but got {threshold}"
    else:
        assert not condition2, f"Weights replacing, expected ratio to be 0.5. but got {ratio}"

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Step1: Loading the model
    modnet = MODNet_auto(backbone_pretrained=False)
    modnet.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(pretrained_path).items()})

    # Step2: Build block
    blocks = get_model_block(modnet)

    # Step3: Get the number of channels after pruning(initial)
    cfg_out_channels = get_pruning_cfg(blocks, ratio=ratio, threshold=threshold)

    # Step4: Further extract cfg for backbone
    cfg_residual_init = []
    cfg_backbone = cfg_out_channels[:nums_of_backbone]
    for i in range(2, len(cfg_backbone), interval_of_residual):
        cfg_residual_init.append(cfg_backbone[i])
    cfg_residual_init = [cfg_backbone[0]] + cfg_residual_init + [cfg_backbone[-1]]

    # Extract expansion and Update residual
    cfg_residual_update = cfg_backbone[fus_layer_feature1:fus_layer_feature17]
    cfg_expansion = [1]
    for i in range(0, len(cfg_residual_update), interval_of_residual):
        scale = int(cfg_residual_update[i + 1] / cfg_residual_update[i])
        cfg_expansion.append(scale)
        cfg_residual_update[i], cfg_residual_update[i + 1], cfg_residual_update[i + 2] = cfg_residual_update[i], int(
            scale * cfg_residual_update[i]), int(
            scale * cfg_residual_update[i])

    # Final cfg output channels for model
    cfg_out_channels = cfg_backbone[:fus_layer_feature1] + cfg_residual_update + cfg_backbone[
                                                                                 fus_layer_feature17:] + cfg_out_channels[
                                                                                                         nums_of_backbone:]
    # Configure the fusion input of the three branches
    new_lr_channels = [cfg_out_channels[fus_layer_lr8x], cfg_out_channels[fus_layer_lr16x]]
    new_hr_channels = [cfg_out_channels[fus_layer_feature1], cfg_out_channels[fus_layer_feature3]]
    new_f_channels = [cfg_out_channels[fus_layer_lr8x]]

    # Configure the inputs of hr and f to build cfg_in_channels
    hr_channels = int(original_channels_of_hr_branch * ratio)
    cfg_hr_in = [new_hr_channels[0], hr_channels + 3, new_hr_channels[1], 2 * hr_channels,
                 2 * hr_channels + original_channels_of_hr_branch + 3,
                 2 * hr_channels, 2 * hr_channels, 2 * hr_channels, 2 * hr_channels, hr_channels, hr_channels,
                 hr_channels + 3, hr_channels]
    cfg_f_in = [new_f_channels[0], 2 * hr_channels, hr_channels + 3, int(hr_channels / 2)]
    cfg_in_channels = [None] * nums_of_lr_branch + cfg_hr_in + cfg_f_in

    # Step5: Build new model
    new_modnet = MODNet_auto(cfg=cfg_residual_init, expansion=cfg_expansion, lr_channel=new_lr_channels,
                             hr_channel=new_hr_channels,
                             f_channel=new_f_channels,
                             hr_channels=int(original_channels_of_hr_branch * (1 - ratio)),
                             backbone_pretrained=False)

    # Calculate the number of parameters and the amount of computation
    dummy_input = torch.randn([1, 3, 512, 512])
    flops, params, _ = count_flops_params(new_modnet, dummy_input, verbose=False)
    print(f"Pruned Model:\nFLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M")

    if weight_replace:
        print("\nStart replacing parameters in the MODNet-P...")
        cfg_mask_model_output, cfg_mask_hr_f_input = model_pruning(modnet, cfg_out_channels, cfg_in_channels)
        pruned_modnet = param_substitution(modnet, new_modnet, cfg_mask_model_output, cfg_mask_hr_f_input, verbose)
        new_modnet = pruned_modnet

        # Calculate the parameters and computational complexity of the pruned model
        flops, params, _ = count_flops_params(pruned_modnet, dummy_input, verbose=False)
        print(f"\nPruned Model after Weight Replacing:\nFLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M")

    # Saving pruned modnet
    save_path = f'{save_dir}modnet_p_ratio_{ratio}_thresh_{threshold}.ckpt'
    print(f"\nSave_model_to={save_path}")
    torch.save(new_modnet.state_dict(), save_path)


if __name__ == '__main__':
    # Define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path of the checkpoint that will be pruned. Here you can load the official model as well as the model trained on our own data.')
    parser.add_argument('--ratio', type=float, default=0.5, required=False,
                        help='Pruning scale, for the filter of hr branch and f branch in MODNet')
    parser.add_argument('--threshold', type=float, default=0.5, required=False,
                        help='Pruning threshold for the adaptive part, for the MobileNetv2 part in the MODNet')
    parser.add_argument('--weight_replace', type=bool, default=False, required=False,
                        help='True or False. It depends on whether to fine-tune or train from scratch after pruning. Warm Tips:It is recommended that the mode can not be enabled, otherwise, try to turn on when the ratio is 0.5 and the threshold is less than 0.6.')
    parser.add_argument('--verbose', type=bool, default=True, required=False,
                        help='Used to print or not the channel changes for each layer of MODNet when weight-place is Ture.')
    args = parser.parse_args()

    # Check input argument
    ckpt_path = args.ckpt_path
    ratio = args.ratio
    threshold = args.threshold
    weight_replace = args.weight_replace
    verbose = args.verbose

    run(ckpt_path, ratio, threshold, weight_replace, verbose)
