<div align="center">
   <a href="https://img.shields.io/badge/Hello-Buddy~-red"><img src="https://img.shields.io/badge/Hello-Buddy~-red.svg"></a>
   <a href="https://img.shields.io/badge/Enjoy-Yourself-brightgreen"><img src="https://img.shields.io/badge/Enjoy-Yourself-brightgreen.svg"></a>
   [简体中文](README.md) | [English](README.EN.md)
  </div>

# Introduction

基于经典L1-Norm评价准则，我们采用了一种自适应与固定比例相结合(Adaptive and Fixed-scale Pruning)的剪枝策略对视频人像抠图模型MODNet进行压缩。该策略较大程度去除了MODNet中的冗余参数，降低了计算代价，在存储资源的利用上节省了79%！✨

---

此外，本项目采用OpenVINO将边缘计算引入视频人像抠图技术，通过边缘端推理测试，剪枝模型MODNet-P取得了一定的速度提升与较好的视觉效果！✨

# Usage

### 1 克隆项目并安装依赖

```bash
git clone https://github.com/sisyphus-cv-lab/MODNet-ModelCompression
cd MODNet-ModelCompression
pip install -r requirements.txt  
```

### 2 下载并解压数据集

```bash
wget -c https://paddleseg.bj.bcebos.com/matting/datasets/PPM-100.zip -O src/datasets/PPM-100.zip
mkdir src/datasets
unzip src/datasets/PPM-100.zip -d src/datasets
```

### 3 下载模型

获取[地址](https://drive.google.com/drive/folders/1SiVFYBkrkokBdv-EGyz1UKjQebgvV2Wy?usp=share_link)，各模型含义如下：

* new_modnet：MODNet官方预训练模型。但为了便于剪枝，网络结构定义发生了轻微改变；
* new_mobilenetv2_human_seg：backbone官方预训练模型，改变同上；
* our_modnet：通过在合成数据集上训练得到，作为剪枝基准模型；
* pruned_modnet：剪枝与再训练后的模型；

### 4 模型剪枝 

```bash
python main_prune.py --ckpt_path .\pretrained\our_modnet.ckpt --ratio 0.5 --threshold 0.5
```

NOTE：

* 使用 "python main.py -h" 获取剪枝时待输入的参数及相关介绍；
* threshold 为控制MODNet主干网络MobileNetV2部分的剪枝阈值；
* ratio 为控制MODNet其他分支的剪枝比例；
* 剪枝完成后，得到剪枝后的模型及其对应的网络配制文件（.json）；该配制文件用于再训练、模型评估、模型推理以及模型导出时网络的构建；

### 5 模型再训练

对步骤4中剪枝得到的模型进行再训练，以恢复精度。

```bash
python .\src\trainer.py --model-path .\result\modnet_p_ratio_0.5_thresh_0.5.ckpt --batch-size 2 --epoch 4
```

NOTE：

* 默认每一个轮次保存模型，以便通过模型评价得到最佳模型；

### 6 模型评价

```bash
python .\src\eval.py --ckpt-path .\result\modnet_p_ratio_0.5_thresh_0.5_epoch0.ckpt --prune-info .\result\modnet_p_ratio_0.5_thresh_0.5.json
```

### 7 模型推理

```bash
python .\src\infer.py --ckpt-path .\result\modnet_p_ratio_0.5_thresh_0.5_epoch0.ckpt --prune-info .\result\modnet_p_ratio_0.5_thresh_0.5.json
```

### 8 模型导出

将再训练后的最优剪枝模型导出，假定epoch0为最优模型，参考如下：

```bash
python .\onnx\export_onnx.py --ckpt-path .\result\modnet_p_ratio_0.5_thresh_0.5_epoch0.ckpt --prune-info .\result\modnet_p_ratio_0.5_thresh_0.5.json --output-path .\result\modnet_p_ratio_0.5_thresh_0.5_best.onnx
```

### 9 模型优化

使用OpenVINO 中的模型优化器model optimizer融合BN，从而实现模型的进一步压缩与加速。

```bash
mo --input_model .\result\modnet_p_ratio_0.5_thresh_0.5_best.onnx --model_name pruned_modnet --output_dir .\result\
```

### 10 MODNet-P 模型推理

通过模型优化得到xml与bin文件后，利用OpenVINO Python API 装载、完成模型推理。

```bash
python inference_openvino.py --model-path .\result\modnet_p_ratio_0.5_thresh_0.5.xml --image-path .\data\img.jpg --device CPU
```

# Results on PPM-100

### 剪枝前后模型对比

| INDEX           | MODNet   | MODNet-P |
| --------------- | -------- | -------- |
| Params（M）     | 6.45     | 1.34     |
| FLOPs（G）      | 18.32    | 4.38     |
| Model size（M） | 25.64    | 5.66     |
| MSE             | 0.009912 | 0.018713 |
| MAD             | 0.013661 | 0.022816 |

NOTE:

* 训练数据集通过随机组合得到，因此，表格中MODNet精度指标MSE、MAD与原论文不一致。

---

### 剪枝前后模型推理速度对比

| Speed on device    | MODNet    | MODNet-P  |
| ------------------ | --------- | --------- |
| Intel i7-8565U CPU | 88.86 ms  | 45.93 ms  |
| NSC2               | 167.93 ms | 101.93 ms |
| ...                | ...       | ...       |

NOTE:

* 使用OpenVINO在NSC2上推理时，需要USB3.0的接口；

---

### 模型再训练方式对比

| Method           | fine-tune | train-from-scratch |
| ---------------- | --------- | ------------------ |
| fixed backbone   | 0.018291  | 0.015588           |
| unfixed backbone | 0.018632  | 0.016826           |

NOTE:

* 进一步对比`微调`与`从头训练`两种方式的性能，我们通过固定主干网络与否对MODNet进行剪枝、测试；

* 为了便于观察比较，这里仅使用MSE作为评价准则。

# Contact

关于本项目任何的疑问、建议，欢迎联系 hbchenstu@outlook.com.

# Reference

https://github.com/ZHKKKe/MODNet

https://github.com/actboy/MODNet

https://github.com/Eric-mingjie/rethinking-network-pruning

https://github.com/kingpeter2015/libovmatting

