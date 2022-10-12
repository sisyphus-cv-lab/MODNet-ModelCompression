from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2 as cv
import os

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
        # print('Infer time:', end - start)

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

        img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
        prd_img = cv.cvtColor(np.asarray(prd_img), cv.COLOR_RGB2BGR)
        res = cv.add(img, 255 - prd_img)
        h, w = img.shape[:2]
        res = cv.resize(res, (int(w / 4), int(h / 4)))
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

        print(ckpt_path)
        infer_images(model, file_dir, save_dir)


if __name__ == '__main__':
    dataset_dir = 'F:/AI/MODNet/src/datasets/PPM-100/val/fg/'

    # # load official modnet
    # modnet = MODNet(backbone_pretrained=False)
    # modnet = nn.DataParallel(modnet)
    #
    # # func 1, infer images with single model
    # ckp_pth = '../pretrained/modnet_photographic_portrait_matting.ckpt'
    # save_dir = './test_result/' + ckp_pth.split('/')[-1][:-5]
    # images_with_single_model(modnet, ckp_pth, dataset_dir, save_dir)

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

    # func 1, infer images with single model(只针对单个模型)
    ckp_pth = '../pretrained/our_model/pruned_modnet.ckpt'
    save_dir = './result/' + ckp_pth.split('/')[-1][:-5]
    images_with_single_model(modnet, ckp_pth, dataset_dir, save_dir)

    # # func 2, infer images with multiple models in dir(针对文件夹中的所有模型)
    # model_dir = '../pretrained/our_model/'
    # save_path = './res/'
    # infer_images_with_models(modnet, model_dir, dataset_dir, save_path)
