

# MODNet Pruning

We adopt a pruning method combining **adaptive and fixed scale** for the video portrait keying model MODNet, based on the **L1-norm pruning** strategy. While reducing the number of model parameters and computational effort, it improves the inference speed and maintains certain visual effects.

## Usage

### 1 Clone and install packages

```bash
git clone https://github.com/sisyphus-cv-lab/MODNet-ModelCompression
cd MODNet-ModelCompression
pip install -r requirements.txt  
```

### 2 Download pretrained model 

### 3 Quick prunning

```bash
python main.py --ckpt_path=./pretrained/our_modnet.ckpt --ratio 0.5 --threshold 0.5
```

Warm tips: Use "python main.py -h" to get the relevant reference when pruning~~

## Results on PPM-100

### Comparison of the models before and after pruning

| INDEX           | MODNet   | MODNet-P |
| --------------- | -------- | -------- |
| Params（M）     | 6.45     | 1.34     |
| FLOPs（G）      | 18.32    | 4.38     |
| Model size（M） | 25.64    | 5.66     |
| MSE             | 0.004318 | 0.018713 |
| MAD             | 0.008163 | 0.022816 |

NOTE:

1. Here we retrained MODNet by adopting our own constructed dataset, so the MSE and MAD in the table are not mentioned in the original paper of MODNet.
2. We used a combination of **Adaptive-Pruning and Fixed-Scale,** where the threshold of adaptive pruning was 0.5 and the scale of fixed-scale was 0.5.

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

## Contact

If you have any questions, please feel free to contact hbchenstu@outlook.com.

## Reference

https://github.com/ZHKKKe/MODNet

https://github.com/actboy/MODNet

https://github.com/Eric-mingjie/rethinking-network-pruning

https://github.com/kingpeter2015/libovmatting

