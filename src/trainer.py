import math
import json
import argparse

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import grey_dilation, grey_erosion

__all__ = [
    'supervised_training_iter',
    'soc_adaptation_iter',
]


# ----------------------------------------------------------------------------------
# Tool Classes/Functions
# ----------------------------------------------------------------------------------

class GaussianBlurLayer(nn.Module):
    """ Add Gaussian Blur to a 4D tensors
    This layer takes a 4D tensor of {N, C, H, W} as input.
    The Gaussian blur will be performed in given channel number (C) splitly.
    """

    def __init__(self, channels, kernel_size):
        """
        Arguments:
            channels (int): Channel for input tensor
            kernel_size (int): Size of the kernel used in blurring
        """

        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0

        self.op = nn.Sequential(
            nn.ReflectionPad2d(math.floor(self.kernel_size / 2)),
            nn.Conv2d(channels, channels, self.kernel_size,
                      stride=1, padding=0, bias=None, groups=channels)
        )

        self._init_kernel()

    def forward(self, x):
        """
        Arguments:
            x (torch.Tensor): input 4D tensor
        Returns:
            torch.Tensor: Blurred version of the input
        """

        if not len(list(x.shape)) == 4:
            print('\'GaussianBlurLayer\' requires a 4D tensor as input\n')
            exit()
        elif not x.shape[1] == self.channels:
            print('In \'GaussianBlurLayer\', the required channel ({0}) is'
                  'not the same as input ({1})\n'.format(self.channels, x.shape[1]))
            exit()

        return self.op(x)

    def _init_kernel(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8

        n = np.zeros((self.kernel_size, self.kernel_size))
        i = math.floor(self.kernel_size / 2)
        n[i, i] = 1
        kernel = scipy.ndimage.gaussian_filter(n, sigma)

        for name, param in self.named_parameters():
            param.data.copy_(torch.from_numpy(kernel))


# ----------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
# MODNet Training Functions
# ----------------------------------------------------------------------------------

blurer = GaussianBlurLayer(1, 3)  # .cuda
if torch.cuda.is_available():
    blurer.cuda()


def supervised_training_iter(
        modnet, optimizer, image, trimap, gt_matte,
        semantic_scale=10.0, detail_scale=10.0, matte_scale=1.0):
    """ Supervised training iteration of MODNet
    This function trains MODNet for one iteration in a labeled dataset.

    Arguments:
        modnet (torch.nn.Module): instance of MODNet
        optimizer (torch.optim.Optimizer): optimizer for supervised training
        image (torch.autograd.Variable): input RGB image
                                         its pixel values should be normalized
        trimap (torch.autograd.Variable): trimap used to calculate the losses
                                          its pixel values can be 0, 0.5, or 1
                                          (foreground=1, background=0, unknown=0.5)
        gt_matte (torch.autograd.Variable): ground truth alpha matte
                                            its pixel values are between [0, 1]
        semantic_scale (float): scale of the semantic loss
                                NOTE: please adjust according to your dataset
        detail_scale (float): scale of the detail loss
                              NOTE: please adjust according to your dataset
        matte_scale (float): scale of the matte loss
                             NOTE: please adjust according to your dataset

    Returns:
        semantic_loss (torch.Tensor): loss of the semantic estimation [Low-Resolution (LR) Branch]
        detail_loss (torch.Tensor): loss of the detail prediction [High-Resolution (HR) Branch]
        matte_loss (torch.Tensor): loss of the semantic-detail fusion [Fusion Branch]

    Example:
        import torch
        from src.models.modnet import MODNet
        from src.trainer import supervised_training_iter

        bs = 16         # batch size
        lr = 0.01       # learn rate
        epochs = 40     # total epochs

        modnet = torch.nn.DataParallel(MODNet()).cuda()
        optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * epochs), gamma=0.1)

        dataloader = CREATE_YOUR_DATALOADER(bs)     # NOTE: please finish this function

        for epoch in range(0, epochs):
            for idx, (image, trimap, gt_matte) in enumerate(dataloader):
                semantic_loss, detail_loss, matte_loss = \
                    supervised_training_iter(modnet, optimizer, image, trimap, gt_matte)
            lr_scheduler.step()
    """

    global blurer

    # set the model to train mode and clear the optimizer
    modnet.train()
    optimizer.zero_grad()

    # forward the model
    pred_semantic, pred_detail, pred_matte = modnet(image, False)

    # calculate the boundary mask from the trimap
    boundaries = (trimap < 0.5) + (trimap > 0.5)

    # calculate the semantic loss
    gt_semantic = F.interpolate(gt_matte, scale_factor=1 / 16, mode='bilinear')
    gt_semantic = blurer(gt_semantic)
    semantic_loss = torch.mean(F.mse_loss(pred_semantic, gt_semantic))
    semantic_loss = semantic_scale * semantic_loss

    # calculate the detail loss
    pred_boundary_detail = torch.where(boundaries, trimap, pred_detail)
    gt_detail = torch.where(boundaries, trimap, gt_matte)
    detail_loss = torch.mean(F.l1_loss(pred_boundary_detail, gt_detail))
    detail_loss = detail_scale * detail_loss

    # calculate the matte loss
    pred_boundary_matte = torch.where(boundaries, trimap, pred_matte)
    matte_l1_loss = F.l1_loss(pred_matte, gt_matte) + 4.0 * F.l1_loss(pred_boundary_matte, gt_matte)
    matte_compositional_loss = F.l1_loss(image * pred_matte, image * gt_matte) \
                               + 4.0 * F.l1_loss(image * pred_boundary_matte, image * gt_matte)
    matte_loss = torch.mean(matte_l1_loss + matte_compositional_loss)
    matte_loss = matte_scale * matte_loss

    # calculate the final loss, backward the loss, and update the model
    loss = semantic_loss + detail_loss + matte_loss
    loss.backward()
    optimizer.step()

    # for test
    return semantic_loss, detail_loss, matte_loss


def soc_adaptation_iter(
        modnet, backup_modnet, optimizer, image,
        soc_semantic_scale=100.0, soc_detail_scale=1.0):
    """ Self-Supervised sub-objective consistency (SOC) adaptation iteration of MODNet
    This function fine-tunes MODNet for one iteration in an unlabeled dataset.
    Note that SOC can only fine-tune a converged MODNet, i.e., MODNet that has been
    trained in a labeled dataset.

    Arguments:
        modnet (torch.nn.Module): instance of MODNet
        backup_modnet (torch.nn.Module): backup of the trained MODNet
        optimizer (torch.optim.Optimizer): optimizer for self-supervised SOC
        image (torch.autograd.Variable): input RGB image
                                         its pixel values should be normalized
        soc_semantic_scale (float): scale of the SOC semantic loss
                                    NOTE: please adjust according to your dataset
        soc_detail_scale (float): scale of the SOC detail loss
                                  NOTE: please adjust according to your dataset

    Returns:
        soc_semantic_loss (torch.Tensor): loss of the semantic SOC
        soc_detail_loss (torch.Tensor): loss of the detail SOC

    Example:
        import copy
        import torch
        from src.models.modnet import MODNet
        from src.trainer import soc_adaptation_iter

        bs = 1          # batch size
        lr = 0.00001    # learn rate
        epochs = 10     # total epochs

        modnet = torch.nn.DataParallel(MODNet()).cuda()
        modnet = LOAD_TRAINED_CKPT()    # NOTE: please finish this function

        optimizer = torch.optim.Adam(modnet.parameters(), lr=lr, betas=(0.9, 0.99))
        dataloader = CREATE_YOUR_DATALOADER(bs)     # NOTE: please finish this function

        for epoch in range(0, epochs):
            backup_modnet = copy.deepcopy(modnet)
            for idx, (image) in enumerate(dataloader):
                soc_semantic_loss, soc_detail_loss = \
                    soc_adaptation_iter(modnet, backup_modnet, optimizer, image)
    """

    global blurer

    # set the backup model to eval mode
    backup_modnet.eval()

    # set the main model to train mode and freeze its norm layers
    modnet.train()
    modnet.module.freeze_norm()

    # clear the optimizer
    optimizer.zero_grad()

    # forward the main model
    pred_semantic, pred_detail, pred_matte = modnet(image, False)

    # forward the backup model
    with torch.no_grad():
        _, pred_backup_detail, pred_backup_matte = backup_modnet(image, False)

    # calculate the boundary mask from `pred_matte` and `pred_semantic`
    pred_matte_fg = (pred_matte.detach() > 0.1).float()
    pred_semantic_fg = (pred_semantic.detach() > 0.1).float()
    pred_semantic_fg = F.interpolate(pred_semantic_fg, scale_factor=16, mode='bilinear')
    pred_fg = pred_matte_fg * pred_semantic_fg

    n, c, h, w = pred_matte.shape
    np_pred_fg = pred_fg.data.cpu().numpy()
    np_boundaries = np.zeros([n, c, h, w])
    for sdx in range(0, n):
        sample_np_boundaries = np_boundaries[sdx, 0, ...]
        sample_np_pred_fg = np_pred_fg[sdx, 0, ...]

        side = int((h + w) / 2 * 0.05)
        dilated = grey_dilation(sample_np_pred_fg, size=(side, side))
        eroded = grey_erosion(sample_np_pred_fg, size=(side, side))

        sample_np_boundaries[np.where(dilated - eroded != 0)] = 1
        np_boundaries[sdx, 0, ...] = sample_np_boundaries

    boundaries = torch.tensor(np_boundaries).float().cuda()

    # sub-objectives consistency between `pred_semantic` and `pred_matte`
    # generate pseudo ground truth for `pred_semantic`
    downsampled_pred_matte = blurer(F.interpolate(pred_matte, scale_factor=1 / 16, mode='bilinear'))
    pseudo_gt_semantic = downsampled_pred_matte.detach()
    pseudo_gt_semantic = pseudo_gt_semantic * (pseudo_gt_semantic > 0.01).float()

    # generate pseudo ground truth for `pred_matte`
    pseudo_gt_matte = pred_semantic.detach()
    pseudo_gt_matte = pseudo_gt_matte * (pseudo_gt_matte > 0.01).float()

    # calculate the SOC semantic loss
    soc_semantic_loss = F.mse_loss(pred_semantic, pseudo_gt_semantic) + F.mse_loss(downsampled_pred_matte,
                                                                                   pseudo_gt_matte)
    soc_semantic_loss = soc_semantic_scale * torch.mean(soc_semantic_loss)

    # NOTE: using the formulas in our paper to calculate the following losses has similar results
    # sub-objectives consistency between `pred_detail` and `pred_backup_detail` (on boundaries only)
    backup_detail_loss = boundaries * F.l1_loss(pred_detail, pred_backup_detail, reduction='none')
    backup_detail_loss = torch.sum(backup_detail_loss, dim=(1, 2, 3)) / torch.sum(boundaries, dim=(1, 2, 3))
    backup_detail_loss = torch.mean(backup_detail_loss)

    # sub-objectives consistency between pred_matte` and `pred_backup_matte` (on boundaries only)
    backup_matte_loss = boundaries * F.l1_loss(pred_matte, pred_backup_matte, reduction='none')
    backup_matte_loss = torch.sum(backup_matte_loss, dim=(1, 2, 3)) / torch.sum(boundaries, dim=(1, 2, 3))
    backup_matte_loss = torch.mean(backup_matte_loss)

    soc_detail_loss = soc_detail_scale * (backup_detail_loss + backup_matte_loss)

    # calculate the final loss, backward the loss, and update the model
    loss = soc_semantic_loss + soc_detail_loss

    loss.backward()
    optimizer.step()

    return soc_semantic_loss, soc_detail_loss


# ----------------------------------------------------------------------------------


if __name__ == '__main__':
    from matting_dataset import MattingDataset, Rescale, \
        ToTensor, Normalize, ToTrainArray, \
        ConvertImageDtype, GenTrimap
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from src.models.modnet_auto import MODNet_auto

    transform = transforms.Compose([
        Rescale(512),
        GenTrimap(),
        ToTensor(),
        ConvertImageDtype(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ToTrainArray()
    ])

    mattingDataset = MattingDataset(transform=transform)

    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True,
                        help='path of the pruned model that will be retrained')
    parser.add_argument('--batch-size', type=int, default=2, required=True,
                        help='BATCH SIZE')
    parser.add_argument('--lr', type=float, default=0.01, required=False,
                        help='LEARNING RATE')
    parser.add_argument('--epoch', type=int, default=2, required=True,
                        help='TOTAL EPOCH ')
    parser.add_argument('--semantic-scale', type=float, default=10.0, required=False,
                        help='SEMANTIC_SCALE')
    parser.add_argument('--detail-scale', type=float, default=10.0, required=False,
                        help='DETAIL_SCALE')
    parser.add_argument('--matte-scale', type=float, default=1.0, required=False,
                        help='MATTE_SCALE')
    parser.add_argument('--save-step', type=int, default=1, required=False,
                        help='SAVE_EPOCH_STEP')
    args = parser.parse_args()

    # check input argumentsd
    model_path = args.model_path
    BS = args.batch_size
    LR = args.lr
    EPOCHS = args.epoch

    SEMANTIC_SCALE = args.semantic_scale
    DETAIL_SCALE = args.detail_scale
    MATTE_SCALE = args.matte_scale

    SAVE_EPOCH_STEP = args.save_step

    prune_info_path = model_path.replace("ckpt", 'json')
    model_name = model_path.replace(".ckpt", "")

    # Get info of pruned model
    prune_info = json.load(open(prune_info_path))
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

    state_dict = torch.load(model_path)
    modnet.load_state_dict(state_dict)
    modnet = torch.nn.DataParallel(modnet)  # .cuda()
    if torch.cuda.is_available():
        modnet = modnet.cuda()

    optimizer = torch.optim.SGD(modnet.parameters(), lr=LR, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * EPOCHS), gamma=0.1)

    dataloader = DataLoader(mattingDataset,
                            batch_size=BS,
                            shuffle=True)

    for epoch in range(0, EPOCHS):
        print(f'epoch: {epoch}/{EPOCHS - 1}')
        for idx, (image, trimap, gt_matte) in enumerate(dataloader):
            semantic_loss, detail_loss, matte_loss = \
                supervised_training_iter(modnet, optimizer, image, trimap, gt_matte,
                                         semantic_scale=SEMANTIC_SCALE,
                                         detail_scale=DETAIL_SCALE,
                                         matte_scale=MATTE_SCALE)
            print(f'{(idx + 1) * BS}/{len(mattingDataset)} --- '
                  f'semantic_loss: {semantic_loss:f}, detail_loss: {detail_loss:f}, matte_loss: {matte_loss:f}\r',
                  end='')
        lr_scheduler.step()

        # 保存中间训练结果
        if epoch % SAVE_EPOCH_STEP == 0:
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': modnet.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': {'semantic_loss': semantic_loss, 'detail_loss': detail_loss, 'matte_loss': matte_loss},
            # }, f'{model_name}_{epoch + 1}_epoch.ckpt')
            torch.save(modnet.state_dict(), f'{model_name}_epoch{epoch}.ckpt')
        print(f'{len(mattingDataset)}/{len(mattingDataset)} --- '
              f'semantic_loss: {semantic_loss:f}, detail_loss: {detail_loss:f}, matte_loss: {matte_loss:f}')
