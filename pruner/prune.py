import numpy as np

np.seterr(divide='ignore', invalid='ignore')


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# 固定比例剪枝
def get_nums_of_keep_channels(sparsity, initial_channels):
    return int((1 - sparsity) * initial_channels) if initial_channels != 1 else 1


# 自适应剪枝，根据threshold计算符合要求的通道
def compute_weights(m, threshold=0.5):
    # weight = m.weight.data.abs().numpy()
    weight = m.abs().numpy()
    L1_norm = np.sum(weight, axis=(1, 2, 3))
    x = sorted(L1_norm, reverse=True)
    normalized = (x - np.min(x)) / (np.max(x) - np.min(x))  # 归一化，用于绘图

    idx = 0
    flag = True
    for filter_idx in range(len(normalized)):
        if normalized[filter_idx] < threshold and flag:
            idx = filter_idx
            flag = False

    per = (idx + 1) / len(normalized)
    if int(per * len(normalized)) != 1:
        keep_nums = _make_divisible(per * len(normalized), 8)
        return keep_nums
    else:
        return 1

