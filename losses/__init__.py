import torch.optim as optim
import torch
# from utils.mri_transforms import complex_abs
from losses.ssim import SSIMLoss
import torch.nn as nn
from losses.hfen import HFENLoss, HFENL1Loss, HFENL2Loss
from losses.nmse import NMSELoss
from losses.nmae import NMAELoss
from torch.nn import functional as F


def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return ((data ** 2 + 1e-8).sum(dim=-1)).sqrt()


def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    elif config.optim.optimizer == 'Adamw':
        return optim.AdamW(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))


def get_loss(output, gt, criterion='mse', ctype=''):
    reduction = "mean"
    if criterion == 'mse':
        func = torch.nn.MSELoss()
        loss_train = func(output, gt.float())
    elif criterion == 'hfen1':
        gt = complex_abs(gt).unsqueeze(1)
        pred = complex_abs(output).unsqueeze(1)
        loss_train = HFENL1Loss(reduction=reduction, norm=True).to(pred.device).forward(pred, gt)
    elif criterion == 'hfen2':
        gt = complex_abs(gt).unsqueeze(1)
        pred = complex_abs(output).unsqueeze(1)
        loss_train = HFENL2Loss(reduction=reduction, norm=True).to(pred.device).forward(pred, gt)
    elif criterion == 'nmse':
        gt = complex_abs(gt).unsqueeze(1)
        pred = complex_abs(output).unsqueeze(1)
        loss_train = NMSELoss(reduction=reduction).to(pred.device).forward(pred, gt)
    elif criterion == 'nmae':
        gt = complex_abs(gt).unsqueeze(1)
        pred = complex_abs(output).unsqueeze(1)
        loss_train = NMAELoss(reduction=reduction).to(pred.device).forward(pred, gt)
    elif criterion == 'ssim':
        gt = complex_abs(gt).unsqueeze(1)
        pred = complex_abs(output).unsqueeze(1)
        data_range = torch.tensor([gt.max()], device=gt.device)
        loss_train = SSIMLoss().to(pred.device).forward(pred, gt, data_range=data_range)
    return loss_train
