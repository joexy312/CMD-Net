import glob
import os
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.fftc import ifftshift, ifft2_2c
from utils.mri_transforms import complex_abs, combine_all_coils
from utils.subsample import MaskFunc
from utils.util import to_tensor


def data_for_training(rawdata, sensitivity, mask_func, norm=True):

    coils, Ny, Nx, ps = rawdata.shape
    shift_kspace = rawdata
    x, y = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1))
    adjust = (-1) ** (x + y)
    shift_kspace = ifftshift(shift_kspace, dim=[-3, -2]) * torch.from_numpy(adjust).view(1, Ny, Nx, 1).float()

    # print(f'shift_kspace shape{shift_kspace.shape}')
    shape = np.array(shift_kspace.shape)  # (15, 640, 368, 2)
    shape[:-3] = 1  # (1, 640, 368, 2)
    mask = mask_func(shape)  # centered
    # print(f'mask {mask.shape}')
    mask = torch.fft.ifftshift(mask)  # NOT centered
    masked_kspace = torch.where(mask == 0, torch.Tensor([0]), shift_kspace)
    masks = mask.repeat(1, Ny, 1, 1)

    img_gt, img_und = ifft2_2c(shift_kspace), ifft2_2c(masked_kspace)

    if norm:
        norm = complex_abs(img_und).max()
        if norm < 1e-6:
            norm = 1e-6
    else:
        norm = 1
    img_gt, img_und = img_gt / norm, img_und / norm
    rawdata_und = masked_kspace / norm  # faster

    sense_gt = combine_all_coils(img_gt, sensitivity)
    sense_und = combine_all_coils(img_und, sensitivity)

    return sense_und, sense_gt, rawdata_und, masks, sensitivity


class MRIDataset(DataLoader):
    def __init__(self, data_list, acceleration, center_fraction):

        self.data_list = data_list
        self.acceleration = acceleration
        self.center_fraction = center_fraction

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        subject_id = self.data_list[idx]
        return get_epoch_batch(
            subject_id,
            self.acceleration,
            self.center_fraction
        )


def get_epoch_batch(subject_id, acc, center_fract):
    ''' get training data '''

    rawdata_name, coil_name = subject_id
    # if 'coronal' in rawdata_name:
    rawdata = np.complex64(loadmat(rawdata_name)['rawdata']).transpose(2, 0, 1)
    sensitivity = np.complex64(loadmat(coil_name)['sensitivities']).transpose(2, 0, 1)
    # else:
    # rawdata = np.complex64(h5.File(rawdata_name)['rawdata']).transpose(2, 0, 1)
    # sensitivity = np.complex64(h5.File(coil_name)['sensitivities'])
    mask_func = MaskFunc(center_fractions=[center_fract], accelerations=[acc])
    rawdata = to_tensor(rawdata)
    sensitivity = to_tensor(sensitivity)

    return data_for_training(rawdata, sensitivity, mask_func)


def load_traindata_path(dataset_dir, debug, name):
    """ Go through each subset (training, validation) under the data directory
    and list the file names and landmarks of the subjects
    """
    train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    validation = [13, 14, 15, 16]
    test = [17, 18, 19, 20]

    which_view = os.path.join(dataset_dir, name)
    data_list = {}
    data_list['train'] = []
    data_list['val'] = []
    data_list['test'] = []

    for k in train:
        subject_id = os.path.join(which_view, str(k))
        n_slice = len(glob.glob('{0}/rawdata*.mat'.format(subject_id)))
        for i in range(2, n_slice + 1):
            raw = '{0}/rawdata{1}.mat'.format(subject_id, i)
            sen = '{0}/espirit{1}.mat'.format(subject_id, i)
            data_list['train'] += [[raw, sen]]

    for k in validation:
        subject_id = os.path.join(which_view, str(k))
        n_slice = len(glob.glob('{0}/rawdata*.mat'.format(subject_id)))
        for i in range(2, n_slice + 1):
            raw = '{0}/rawdata{1}.mat'.format(subject_id, i)
            sen = '{0}/espirit{1}.mat'.format(subject_id, i)
            data_list['val'] += [[raw, sen]]

    for k in test:
        subject_id = os.path.join(which_view, str(k))
        n_slice = len(glob.glob('{0}/rawdata*.mat'.format(subject_id)))
        for i in range(2, n_slice + 1):
            raw = '{0}/rawdata{1}.mat'.format(subject_id, i)
            sen = '{0}/espirit{1}.mat'.format(subject_id, i)
            data_list['test'] += [[raw, sen]]

    return data_list


if __name__ == '__main__':

    import visdom
    # vis = visdom.Visdom()  # python -m visdom.server
    # assert vis.check_connection()

    name = 'coronal_pd'
    dataset_dir = '/home/joe/codes/data/knee/'
    data_list = load_traindata_path(dataset_dir, False, name)

    train_set = data_list['train']
    val_set = data_list['val']
    test_set = data_list['test']

    sample_rawdata_name, coil_name = train_set[1]
    rawdata = np.complex64(loadmat(sample_rawdata_name)['rawdata']).transpose(2, 0, 1)
    sensitivity = np.complex64(loadmat(coil_name)['sensitivities']).transpose(2, 0, 1)
    print(f'sequence:{name}, size:{rawdata.shape[1], rawdata.shape[2]}')

    # win = vis.image(np.ones((2, rawdata.shape[1], rawdata.shape[2])), opts=dict(title=name))

    for i in range(len(train_set)):
        rawdata_name, coil_name = train_set[i]
        rawdata = np.complex64(loadmat(rawdata_name)['rawdata']).transpose(2, 0, 1)
        sensitivity = np.complex64(loadmat(coil_name)['sensitivities'])
        mask_func = MaskFunc(center_fractions=[0.08], accelerations=[4])
        rawdata = to_tensor(rawdata)
        sensitivity = to_tensor(sensitivity.transpose(2, 0, 1))

        sense_und, sense_gt, rawdata_und, masks, sensitivity = data_for_training(rawdata, sensitivity, mask_func)

        rawdata_und = torch.fft.ifftshift(rawdata_und)
        plt_org = complex_abs(sense_gt).data.to('cpu').numpy()
        plt_atb = complex_abs(sense_und).data.to('cpu').numpy()
        for i in range(15):
            plt_k = complex_abs(rawdata_und[i]).data.to('cpu').numpy()
            a, b, l = 160, 24, 320
            # A = plt_org / plt_org.max()
            # B = plt_atb / plt_atb.max()
            plt.imshow(plt_k[a:a+l, b:b+l], cmap=plt.cm.gray, clim=(0.0, 0.05))

            # plt.imshow(np.c_[A, B], cmap=plt.cm.gray)
            # plt.imshow(np.c_[A, B], cmap=plt.cm.gray)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=.0)
            plt.title(rawdata_name)
            plt.show()

        # vis.matplot(plt, win=win, opts=dict(title=rawdata_name))