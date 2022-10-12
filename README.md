

# MODNet Pruning

This project  applies the pruning strategy of **L1-norm-pruning** for MODNet.

---

## Quick Start

### Clone and install packages

```bash
git clone https://github.com/sisyphus-cv-lab/MODNet-ModelCompression
cd MODNet-ModelCompression
pip install -r requirements.txt  
```

### Prepare dataset

```bash
wget -c https://paddleseg.bj.bcebos.com/matting/datasets/PPM-100.zip -O src/datasets/PPM-100.zip
unzip src/datasets/PPM-100.zip -d src/datasets
```

---

## Results on PPM-100

Comparison of the official model and the pruned model

| INDEX           | MODNet   | Ours(Prunded) |
| --------------- | -------- | ------------- |
| Params（M）     | 6.48     | 1.36          |
| FLOPs（G）      | 18.32    | 4.5           |
| Model size（M） | 25.64    | 5.67          |
| MSE             | 0.00991  | 0.018713      |
| MAD             | 0.013661 | 0.022816      |

NOTE:

1. Here we retrained MODNet based on the pre-trained weights of MobileNetV2, so the MSE and MAD in the table are not mentioned in the paper.
2. We used a combination of **Adaptive-Pruning and Fixed-Scale,** where the threshold of adaptive pruning was 0.6 and the scale of fixed-scale was 0.5.

---

Comparison of speeds on MODNet and Ours

| Speed on device    | MODNet    | Ours(Prunded) |
| ------------------ | --------- | ------------- |
| Intel i7-8565U CPU | 88.86 ms  | 45.93 ms      |
| NSC2               | 167.93 ms | 101.93 ms     |
| ...                | ...       | ...           |

NOTE:

1. The USB 3.0 interface is required for testing on NSC2;

---

Comparison of the fine-tune and train-from-scratch

|                  | fine-tune | train-from-scratch |
| ---------------- | --------- | ------------------ |
| fixed backbone   | 0.018291  | 0.015588           |
| unfixed backbone | 0.018632  | 0.016826           |

NOTE:

1. Only use MSE as assessment criteria;
2. We pruned the MODNet backbone（mobilenetv2） network whether it was fixed or not;

## Reference

https://github.com/ZHKKKe/MODNet

https://github.com/actboy/MODNet

https://github.com/Eric-mingjie/rethinking-network-pruning

https://github.com/kingpeter2015/libovmatting