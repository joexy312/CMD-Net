import torch
import numpy as np
from torch.autograd import Variable
import argparse
from utils.mri_transforms import complex_abs


def to_tensor(data):
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data).float()


def to_var(tensor):
    return Variable(tensor.cuda()) if torch.cuda.is_available() else tensor


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def normalize01(img):
    """
    Normalize the image between o and 1
    """
    if len(img.shape) == 3:
        nimg = len(img)
    else:
        nimg = 1
        r, c = img.shape
        img = np.reshape(img, (nimg, r, c))
    img2 = np.empty(img.shape, dtype=img.dtype)
    for i in range(nimg):
        # img2[i] = div0(img[i] - img[i].min(), img[i].ptp())
        img2[i]=(img[i]-img[i].min())/(img[i].max()-img[i].min())
    return np.squeeze(img2).astype(img.dtype)


def tensor_to_vis(t):
    t = complex_abs(t).data.to('cpu').numpy().squeeze()
    t /= t.max()
    return t
