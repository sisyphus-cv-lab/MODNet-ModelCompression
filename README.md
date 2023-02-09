

<div align="center">
[简体中文](README.md) | [English](README.EN.md)
<br>

## Introduction

基于经典L1-Norm评价准则，我们采用了一种自适应与固定比例相结合(Adaptive and Fixed-scale Pruning)的剪枝策略对视频人像抠图模型MODNet进行压缩。该策略较大程度去除了MODNet中的冗余参数，并降低了计算代价，在存储资源的利用上节省了79%！✨

---

此外，本项目采用OpenVINO将边缘计算引入视频人像抠图技术，通过边缘端推理测试，剪枝模型MODNet-P取得了一定的速度提升与较好的视觉效果！✨

## Usage

### 1 克隆项目并安装环境依赖

```bash
git clone https://github.com/sisyphus-cv-lab/MODNet-ModelCompression
cd MODNet-ModelCompression
pip install -r requirements.txt  
```

### 2 下载预训练模型

从这里[下载](https://drive.google.com/drive/folders/1SiVFYBkrkokBdv-EGyz1UKjQebgvV2Wy?usp=share_link)。

### 3 剪枝

```bash
python main.py --ckpt_path=./pretrained/new_modnet.ckpt --ratio 0.5 --threshold 0.5
```

Tips: 使用 "python main.py -h" 获取剪枝时指定的参数介绍。

## Results on PPM-100

### 剪枝前后模型对比

| INDEX           | MODNet   | MODNet-P |
| --------------- | -------- | -------- |
| Params（M）     | 6.45     | 1.34     |
| FLOPs（G）      | 18.32    | 4.38     |
| Model size（M） | 25.64    | 5.66     |
| MSE             | 0.009912 | 0.018713 |
| MAD             | 0.013661 | 0.022816 |

NOTE:

1. 训练数据集通过随机组合得到，因此，表格中MODNet精度指标MSE、MAD与原论文不一致。

---

### 剪枝前后模型推理速度对比

| Speed on device    | MODNet    | MODNet-P  |
| ------------------ | --------- | --------- |
| Intel i7-8565U CPU | 88.86 ms  | 45.93 ms  |
| NSC2               | 167.93 ms | 101.93 ms |
| ...                | ...       | ...       |

NOTE:

1. 使用OpenVINO在NSC2上推理时，需要USB3.0的接口；

---

### 模型再训练方式对比

| Method           | fine-tune | train-from-scratch |
| ---------------- | --------- | ------------------ |
| fixed backbone   | 0.018291  | 0.015588           |
| unfixed backbone | 0.018632  | 0.016826           |

NOTE:

2. 进一步对比`微调`与`从头训练`两种方式的性能，我们通过固定主干网络与否对MODNet进行剪枝、测试；
2. 为了便于观察比较，这里仅使用MSE作为评价准则。

## Contact

关于本项目任何的疑问、建议，欢迎联系 hbchenstu@outlook.com.

## Reference

https://github.com/ZHKKKe/MODNet

https://github.com/actboy/MODNet

https://github.com/Eric-mingjie/rethinking-network-pruning

https://github.com/kingpeter2015/libovmatting

