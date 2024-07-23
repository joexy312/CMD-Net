import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from utils.mri_transforms import expand_operator, reduce_operator
from utils.fftc import fft2_2c, ifft2_2c
from models.backbones.unet_2d import UnetModel2d


class dataConsistencyTerm(nn.Module):

    def __init__(self, noise_lvl=None):
        super(dataConsistencyTerm, self).__init__()
        self.noise_lvl = noise_lvl

    def perform(self, x, k0, mask, sensitivity):
        """
        k    - input in k-space
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
        x = expand_operator(x, sensitivity)
        k = fft2_2c(x, (2, 3))

        # k0 = fft2_2c(expand_operator(k0, sensitivity), (2, 3))

        v = self.noise_lvl
        if v is not None:
            out = (1 - mask) * k + mask * (v * k + (1 - v) * k0)
        else:
            out = (1 - mask) * k + mask * k0

        x = ifft2_2c(out, (2, 3))
        x = reduce_operator(x, sensitivity)

        return x


class gd(nn.Module):
    def __init__(self):
        super(gd, self).__init__()
        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

    def _forward_operator(self, image, sampling_mask, sensitivity_map):  # PFS
        forward = torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=image.dtype).to(image.device),
            fft2_2c(expand_operator(image, sensitivity_map, self._coil_dim), dim=self._spatial_dims),
        )
        return forward

    def _backward_operator(self, kspace, sampling_mask, sensitivity_map):  # (PFS)^(-1)
        backward = reduce_operator(
            ifft2_2c(
                torch.where(
                    sampling_mask == 0,
                    torch.tensor([0.0], dtype=kspace.dtype).to(kspace.device),
                    kspace,
                ),
                self._spatial_dims,
            ),
            sensitivity_map,
            self._coil_dim,
        )
        return backward

    def forward(self, x, atb_k, mask, csm):
        Ax = self._forward_operator(x, mask, csm)
        ATAx_y = self._backward_operator(Ax - atb_k, mask, csm)
        r = x - self.lambda_step * ATAx_y

        return r


class var_block(nn.Module):
    def __init__(self):
        super(var_block, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.cnn1 = UnetModel2d(
                    in_channels=2,
                    out_channels=2,
                    num_filters=16,
                    num_pool_layers=4,
                    dropout_probability=0.0,
                )

        self.gd_block = gd()

    def forward(self, x, atb_k, mask, csm):

        r = self.cnn1(x)
        x = self.gd_block(x, atb_k, mask, csm) + 2 * r * self.lambda_step

        return x


class sparse_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.gd_block = gd()

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=(3, 3), padding=1)
        )

        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

    def forward(self, x, atb_k, mask, csm):
        r = self.gd_block(x, atb_k, mask, csm)

        r = r.permute(0, 3, 1, 2)
        identity = r
        xf = self.conv1(r)
        x_soft = torch.mul(torch.sign(xf), F.relu(torch.abs(xf) - self.soft_thr))
        xb = self.conv2(x_soft)
        x = xb.permute(0, 2, 3, 1)

        est = self.conv2(xf)  # self.conv2(self.conv1(r))
        loss = est - identity  # self.conv2(self.conv1(r)) - r

        return x, loss


class cmdNet(nn.Module):
    def __init__(self, iterations=10):
        super().__init__()
        self.iterations = iterations

        self.var_blocks = nn.ModuleList()
        self.sparse_blocks = nn.ModuleList()

        self.attention = nn.ModuleList()

        self.conv_blocks = nn.ModuleList()
        self.dc_blocks = nn.ModuleList()

        alfa = nn.Parameter(torch.Tensor([0.5]))

        for i in range(iterations):
            self.var_blocks.append(var_block())
            self.sparse_blocks.append(sparse_block())
            if i == 0:
                self.attention.append(nn.Sequential(
                    nn.Conv2d(6, 32, (3, 3), padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 2, (3, 3), padding=1),
                    nn.Sigmoid()
                ))
            # elif i == 1:
            #     self.attention.append(nn.Sequential(
            #         nn.Conv2d(10, 64, (3, 3), padding=1),
            #         nn.ReLU(),
            #         nn.Conv2d(64, 2, (3, 3), padding=1),
            #         nn.Sigmoid()
            #     ))
            # elif i == 2:
            #     self.attention.append(nn.Sequential(
            #         nn.Conv2d(14, 64, (3, 3), padding=1),
            #         nn.ReLU(),
            #         nn.Conv2d(64, 2, (3, 3), padding=1),
            #         nn.Sigmoid()
            #     ))
            else:
                self.attention.append(nn.Sequential(
                    nn.Conv2d(10, 32, (3, 3), padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 2, (3, 3), padding=1),
                    nn.Sigmoid()
                ))

            # self.conv_blocks.append(conv.Conv2d(2, 2, 32, 3, nn.ReLU()))
            self.conv_blocks.append(UnetModel2d(
                    in_channels=2,
                    out_channels=2,
                    num_filters=4,
                    num_pool_layers=4,
                    dropout_probability=0.0,
                ))
            self.dc_blocks.append(dataConsistencyTerm(alfa))

    def forward(self, x, atb_k, mask, csm):

        res = []
        losses = []

        x1tmps = []
        x2tmps = []

        initial = x.permute(0, 3, 1, 2).clone().detach()

        for i in range(self.iterations):
            x1 = self.var_blocks[i](x, atb_k, mask, csm)

            x2, loss = self.sparse_blocks[i](x, atb_k, mask, csm)

            losses.append(loss)

            x1_tmp = x1.permute(0, 3, 1, 2)
            x2_tmp = x2.permute(0, 3, 1, 2)
            tmp = torch.cat((x1_tmp, x2_tmp), dim=1)
            x1tmps.append(x1_tmp)
            x2tmps.append(x2_tmp)

            # if i == 0:
            #     att = self.attention[i](torch.cat((tmp, initial), dim=1))
            # else:
            #     att = self.attention[i](torch.cat((tmp, last_tmp, initial), dim=1))
            #
            att = self.attention[i](torch.cat((*x1tmps[-2:], *x2tmps[-2:], initial), dim=1))
            # att = self.attention[i](tmp)
            # last_tmp = tmp
            #
            att1 = att[:, 0, :, :]
            att1 = att1.unsqueeze(1)
            att2 = att[:, 1, :, :]
            att2 = att2.unsqueeze(1)
            att1 = att1.permute(0, 2, 3, 1).squeeze(-1)
            att2 = att2.permute(0, 2, 3, 1).squeeze(-1)
            #
            x1 = torch.view_as_complex(x1)
            x2 = torch.view_as_complex(x2)

            x = (1 - att1) * x2 + att1 * x1
            x = torch.view_as_real(x.squeeze(1))

            new_x = self.conv_blocks[i](x) + x  # residual
            x = att2 * torch.view_as_complex(new_x) + (1 - att2) * torch.view_as_complex(x)
            x = torch.view_as_real(x)
            x = self.dc_blocks[i].perform(x, atb_k, mask, csm)

            res.append(x)

        return res, losses


if __name__ == "__main__":

    model = cmdNet(iterations=10)
    # model_structure(model)
    from thop import profile
    input1 = torch.randn(1, 640, 368, 2)
    input2 = torch.randn(1, 640, 368, 2)
    input3 = torch.randn(1, 15, 640, 368, 2)
    input4 = torch.randn(1, 1, 640, 368, 1)
    input5 = torch.randn(1, 15, 640, 368, 2)
    flops, params = profile(model, (input2, input3, input4, input5))
    print('flops: ', flops, 'params: ', params)