<div align="center">
  <p>
   <a href="https://img.shields.io/badge/Hello-Buddy~-red"><img src="https://img.shields.io/badge/Hello-Buddy~-red.svg"></a>
   <a href="https://img.shields.io/badge/Enjoy-Yourself-brightgreen"><img src="https://img.shields.io/badge/Enjoy-Yourself-brightgreen.svg"></a>
  </p>

[简体中文](README.md) | [English](README.EN.md)
<br>
</div>

# Introduction

We adopt a heuristic pruning criteria combining **adaptive and fixed scale** for the video portrait keying model MODNet, based on the **L1-norm pruning** strategy. This strategy largely eliminates the redundant parameters in MODNet, reduces the computational cost, and saves 79% in the utilization of storage resources! ✨

---

In this project, OpenVINO is used to introduce edge computing into video portrait matting technology, and Our pruning model, that is MODNet-P has achieved certain speed improvement and better visual effects when inferencing on edge devices! ✨

# Usage

### 1 Clone and install packages

```bash
git clone https://github.com/sisyphus-cv-lab/MODNet-ModelCompression
cd MODNet-ModelCompression
pip install -r requirements.txt 
```

### 2 Download and unzip the dataset

```bash
wget -c https://paddleseg.bj.bcebos.com/matting/datasets/PPM-100.zip -O src/datasets/PPM-100.zip
mkdir src/datasets
unzip src/datasets/PPM-100.zip -d src/datasets
```

### 3 Download model

You can download from this [link](https://drive.google.com/drive/folders/1SiVFYBkrkokBdv-EGyz1UKjQebgvV2Wy). The meaning of model as follows:

* new_modnet：`Official pre-trained model`, and for pruning, we made a simple modification to its model definition section.
* new_mobilenetv2_human_seg： `Official backbone pre-trained model`, and same as above for it.
* our modnet：Trained on synthetic dataset and used as the pruning benchmark model.
* pruned_modnet：Pruned and retrained model;

### 4 Model Pruning

```bash
python main_prune.py --ckpt_path .\pretrained\our_modnet.ckpt --ratio 0.5 --threshold 0.5
```

NOTE：

* Use "python main.py -h" to get the relevant reference when pruning！
* threshold is to control the pruning threshold of MobileNetV2 part of MODNet backbone network.
* ratio is the pruning ratio controlling other branches（hr&f branch） of MODNet;
* After the pruning, the pruned model and its corresponding network configuration file (.json) were obtained. (The configuration file was used for retraining, model evaluation, model inference, and model export)

### 5 Retraining

The model obtained by pruning in Step 4 is retrained to restore the accuracy.

```bash
python .\src\trainer.py --model-path .\result\modnet_p_ratio_0.5_thresh_0.5.ckpt --batch-size 2 --epoch 4
```

NOTE：

* By default, the model is saved in each round, so that the best model can be obtained through model evaluation.

### 6 Evaluate

```bash
python .\src\eval.py --ckpt-path .\result\modnet_p_ratio_0.5_thresh_0.5_epoch0.ckpt --prune-info .\result\modnet_p_ratio_0.5_thresh_0.5.json
```

### 7 Inference

```bash
python .\src\infer.py --ckpt-path .\result\modnet_p_ratio_0.5_thresh_0.5_epoch0.ckpt --prune-info .\result\modnet_p_ratio_0.5_thresh_0.5.json
```

### 8 Export ONNX

For example, epoch0 is assumed to be the best model from retraining. The reference is as follows:

```bash
python .\onnx\export_onnx.py --ckpt-path .\result\modnet_p_ratio_0.5_thresh_0.5_epoch0.ckpt --prune-info .\result\modnet_p_ratio_0.5_thresh_0.5.json --output-path .\result\modnet_p_ratio_0.5_thresh_0.5_best.onnx
```

### 9 Model optimization

We use Model Optimizer supported by OpenVINO to fuse the BN and other optimization, for achieving further compression and acceleration of the pruned model.

```bash
mo --input_model .\result\modnet_p_ratio_0.5_thresh_0.5_best.onnx --model_name pruned_modnet --output_dir .\result\
```

### 10 Inference of MODNet-P with OpenVINO

After getting **xml and bin** files through Model Optimization, OpenVINO Python API is used to load and complete model inference.

```bash
python inference_openvino.py --model-path .\result\pruned_modnet.xml --image-path .\data\img.jpg --device CPU
```

# Results on PPM-100

### Comparison of the models before and after pruning

| INDEX           | MODNet   | MODNet-P |
| --------------- | -------- | -------- |
| Params（M）     | 6.45     | 1.34     |
| FLOPs（G）      | 18.32    | 4.38     |
| Model size（M） | 25.64    | 5.66     |
| MSE             | 0.009912 | 0.018713 |
| MAD             | 0.013661 | 0.022816 |

NOTE:

1. Here we retrained MODNet by adopting our own constructed dataset, so the MSE and MAD in the table are not mentioned in the original paper of MODNet.

---

### Comparison of speeds on MODNet and Ours

| Speed on device    | MODNet    | MODNet-P  |
| ------------------ | --------- | --------- |
| Intel i7-8565U CPU | 88.86 ms  | 45.93 ms  |
| NSC2               | 167.93 ms | 101.93 ms |
| ...                | ...       | ...       |

NOTE:

1. The USB 3.0 interface is required for testing on NSC2;

---

### Comparison of performance between fine-tune and train-from-scratch

|                  | fine-tune | train-from-scratch |
| ---------------- | --------- | ------------------ |
| fixed backbone   | 0.018291  | 0.015588           |
| unfixed backbone | 0.018632  | 0.016826           |

NOTE:

2. In order to illustrate the effect of fine-tune and train from scratch after pruning, we pruned and tested the backbone MobileNetV2 in MODNet fixed or not. 
2. Using only MSE as an evaluation criteria;

# Contact

If you have any questions, please feel free to contact hbchenstu@outlook.com.

# Reference

https://github.com/ZHKKKe/MODNet

https://github.com/actboy/MODNet

https://github.com/Eric-mingjie/rethinking-network-pruning

https://github.com/kingpeter2015/libovmatting

