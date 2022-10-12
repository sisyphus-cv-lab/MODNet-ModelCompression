import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from glob import glob

from src.models.modnet import MODNet
from src.models.modnet_auto import MODNet_auto

from infer import predit_matte


def cal_mad(pred, gt):
    diff = pred - gt
    diff = np.abs(diff)
    mad = np.mean(diff)
    return mad


def cal_mse(pred, gt):
    diff = pred - gt
    diff = diff ** 2
    mse = np.mean(diff)
    return mse


def load_eval_dataset(dataset_root_dir='src/datasets/PPM-100'):
    image_path = dataset_root_dir + '/val/fg/*'
    matte_path = dataset_root_dir + '/val/alpha/*'
    image_file_name_list = glob(image_path)
    image_file_name_list = sorted(image_file_name_list)
    matte_file_name_list = glob(matte_path)
    matte_file_name_list = sorted(matte_file_name_list)

    return image_file_name_list, matte_file_name_list


def eval(modnet: MODNet, dataset):
    mse = total_mse = 0.0
    mad = total_mad = 0.0
    cnt = 0

    for im_pth, mt_pth in zip(dataset[0], dataset[1]):
        im = Image.open(im_pth)
        pd_matte = predit_matte(modnet, im)

        gt_matte = Image.open(mt_pth)
        gt_matte = np.asarray(gt_matte) / 255

        total_mse += cal_mse(pd_matte, gt_matte)
        total_mad += cal_mad(pd_matte, gt_matte)

        cnt += 1
    if cnt > 0:
        mse = total_mse / cnt
        mad = total_mad / cnt

    return mse, mad


if __name__ == '__main__':
    dataset_dir = load_eval_dataset('../src/datasets/PPM-100')

    # load official modnet
    # modnet = MODNet(backbone_pretrained=False)
    # modnet = nn.DataParallel(modnet)
    # ckp_pth = '../pretrained/modnet_photographic_portrait_matting.ckpt'

    # load our own trained modnet
    # modnet = MODNet_auto(backbone_pretrained=False)
    # modnet = nn.DataParallel(modnet)
    # ckp_pth = '../pretrained/our_modnet.ckpt'

    # load our pruned model
    ratio = 0.5
    my_cfg = [8, 8, 8, 16, 24, 16, 16, 48, 16, 16, 8, 32, 16, 32, 104, 32, 64, 232, 584]
    my_expansion_cfg = [None, 1, 9, 14, 4, 7, 9, 3, 7, 19, 22, 24, 15, 30, 17, 7, 26, 15, None]
    my_lr_channels = [32, 16]
    my_hr_channels = [8, 16]
    my_f_channels = [32]

    modnet = MODNet_auto(cfg=my_cfg, expansion=my_expansion_cfg, lr_channel=my_lr_channels,
                         hr_channel=my_hr_channels,
                         f_channel=my_f_channels,
                         hr_channels=int(32 * (1 - ratio)),
                         backbone_pretrained=False)

    modnet = nn.DataParallel(modnet)
    ckp_pth = '../pretrained/our_model/pruned_modnet.ckpt'

    # assess 1, infer images with single model
    weights = torch.load(ckp_pth, map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)
    mse, mad = eval(modnet, dataset_dir)
    print(f' mse: {mse:6f}, mad: {mad:6f}')

    # # assess 2, infer images with multiple models in dir
    # # to find the best model with the lowest mse&mad
    # ckp_dir = '../pretrained/12/'
    # ckp_pth_set = os.listdir(ckp_dir)
    # ckp_pth_set.sort(key=lambda x: x[-6:-4])
    # for ckp_name in ckp_pth_set:
    #     ckp_pth = ckp_dir + ckp_name
    #     weights = torch.load(ckp_pth, map_location=torch.device('cpu'))
    #     modnet.load_state_dict(weights)
    #     mse, mad = eval(modnet, dataset_dir)
    #     print(f'{ckp_name}    mse: {mse:6f}, mad: {mad:6f}')
