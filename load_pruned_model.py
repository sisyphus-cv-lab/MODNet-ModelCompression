from src.models.modnet_auto import MODNet_auto
import torch

# 根据剪枝时的参数配置进行手动构建网络
# 再训练时如果基于权重，需要加载；反之，直接从头训练即可
ratio = 0.5
my_cfg = [8, 8, 8, 16, 24, 16, 16, 48, 16, 16, 8, 32, 16, 32, 104, 32, 64, 232, 584]
my_expansion_cfg = [None, 1, 9, 14, 4, 7, 9, 3, 7, 19, 22, 24, 15, 30, 17, 7, 26, 15, None]
my_lr_channels = [32, 16]
my_hr_channels = [8, 16]
my_f_channels = [32]

model = MODNet_auto(cfg=my_cfg, expansion=my_expansion_cfg, lr_channel=my_lr_channels,
                    hr_channel=my_hr_channels,
                    f_channel=my_f_channels,
                    hr_channels=int(32 * (1 - ratio)),
                    backbone_pretrained=False)


dummy_input = torch.randn([1, 3, 512, 512])
model(dummy_input)

# from nni.compression.pytorch.pruner import count_flops_params
# flops, params, _ = count_flops_params(model, dummy_input, verbose=True)
# print(f"Model FLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M")
