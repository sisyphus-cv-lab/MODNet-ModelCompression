"""
# load official modnet
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)
"""
import argparse

from PIL import Image
from time import time

import numpy as np
from torchvision import transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2 as cv
import os
import json

from src.models.modnet import MODNet
from src.models.modnet_auto import MODNet_auto


def predit_matte(modnet: MODNet, im: Image):
    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # define hyper-parameters
    ref_size = 512
    modnet.eval()
    with torch.no_grad():
        # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        im = Image.fromarray(im)
        # convert image to PyTorch tensor
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        # start = time()
        # _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)
        _, _, matte = modnet(im)
        # end = time()
        # print(f'Infer time: {end - start:.6}s')

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        return matte


def pred_result(img, matte):
    prd_img = Image.fromarray(((matte * 255).astype('uint8')), mode='L')

    img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
    prd_img = cv.cvtColor(np.asarray(prd_img), cv.COLOR_RGB2BGR)
    res = cv.add(img, 255 - prd_img)
    h, w = img.shape[:2]
    res = cv.resize(res, (int(w / 4), int(h / 4)))
    cv.imshow('img', res)
    cv.waitKey(0)
    cv.destroyAllWindows()


def infer_images(model, file_dir, save_dir):
    files = list(os.walk(file_dir))[0][2:][0]

    for file_name in files:
        img = Image.open(file_dir + file_name)

        matte = predit_matte(model, img)

        prd_img = Image.fromarray(((matte * 255).astype('uint8')), mode='L')
        prd_img = np.asarray(prd_img)

        img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
        mask = cv.merge([prd_img, prd_img, prd_img])
        res = cv.add(img, 255 - mask)
        h, w = img.shape[:2]
        cv.imwrite(save_dir + "/" + file_name, res)


def images_with_single_model(model, model_path, file_dir, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    weights = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(weights)

    infer_images(model, file_dir, save_path)


def infer_images_with_models(model, model_dir_path, file_dir, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filenames = os.listdir(model_dir_path)
    for filename in filenames:
        ckpt_path = model_dir_path + filename
        weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(weights)

        save_dir = save_path + ckpt_path.split('/')[-1][:-4]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        infer_images(model, file_dir, save_dir)


if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=True,
                        help='path of the checkpoint that will be infered')
    parser.add_argument('--prune-info', type=str, required=True,
                        help='path of the prune info')
    args = parser.parse_args()

    # check input argumentsd
    ckpt_path = args.ckpt_path
    prune_info = args.prune_info

    # load ppm-100 dataset
    dataset_dir = './src/datasets/PPM-100/val/fg/'

    # load our pruned model
    prune_info = json.load(open(prune_info))
    ratio = prune_info['ratio']
    threshold = prune_info['threshold']
    my_cfg = prune_info['new_cfg']
    my_expansion_cfg = prune_info['new_expansion_cfg']
    my_hr_channels = prune_info['new_hr_channels']
    my_lr_channels = prune_info['new_lr_channels']
    my_f_channels = prune_info['new_f_channels']

    modnet = MODNet_auto(cfg=my_cfg, expansion=my_expansion_cfg, lr_channel=my_lr_channels,
                         hr_channel=my_hr_channels,
                         f_channel=my_f_channels,
                         hr_channels=int(32 * (1 - ratio)),
                         backbone_pretrained=False)

    modnet = nn.DataParallel(modnet)

    # func 1, infer images with single model
    save_dir = ckpt_path.split('/')[-1][:-5]
    print(f"Starting......")
    images_with_single_model(modnet, ckpt_path, dataset_dir, save_dir)
    print(f"Save infer result to {save_dir}")

    # # func 2, infer images with multiple models in dir
    # model_dir = '../pretrained/our_model/'
    # save_path = './res/'
    # infer_images_with_models(modnet, model_dir, dataset_dir, save_path)
