import os
import json
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

import auto_modnet_onnx

if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=True,
                        help='path of the checkpoint that will be converted')
    parser.add_argument('--output-path', type=str, required=True,
                        help='path for saving the ONNX model')
    parser.add_argument('--prune-info', type=str, required=True,
                        help='path of the prune info')
    args = parser.parse_args()

    # check input argumentsd
    ckpt_path = args.ckpt_path
    output_path = args.output_path
    prune_info = args.prune_info

    if not os.path.exists(ckpt_path):
        print(f'Cannot find checkpoint path: {ckpt_path}')
        exit()

    # Get info of pruned model
    prune_info = json.load(open(prune_info))
    ratio = prune_info['ratio']
    threshold = prune_info['threshold']
    my_cfg = prune_info['new_cfg']
    my_expansion_cfg = prune_info['new_expansion_cfg']
    my_hr_channels = prune_info['new_hr_channels']
    my_lr_channels = prune_info['new_lr_channels']
    my_f_channels = prune_info['new_f_channels']

    modnet = auto_modnet_onnx.MODNet_auto(cfg=my_cfg, expansion=my_expansion_cfg, lr_channel=my_lr_channels,
                                          hr_channel=my_hr_channels,
                                          f_channel=my_f_channels,
                                          hr_channels=int(32 * (1 - ratio)),
                                          backbone_pretrained=False)

    # define model & load checkpoint
    modnet = nn.DataParallel(modnet).cpu()
    state_dict = torch.load(ckpt_path)
    modnet.load_state_dict(state_dict)
    modnet.eval()

    # prepare dummy_input
    batch_size = 1
    height = 512
    width = 512
    dummy_input = Variable(torch.randn(batch_size, 3, height, width)).cpu()

    # export to onnx model
    torch.onnx.export(
        modnet.module, dummy_input, output_path, export_params=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                      'output': {0: 'batch_size', 2: 'height', 3: 'width'}}, opset_version=11)

    print("Successfully Exporting ONNX model")
    print(f"Saving to {output_path}")
