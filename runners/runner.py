from datasets import get_dataset
from losses import get_optimizer, get_loss
from tqdm import tqdm
from utils.util import to_var
from utils.mri_transforms import complex_abs
from utils.metrics import psnr, ssim, nmse
from utils.util import tensor_to_vis
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from models.cmdNet import cmdNet


def update_pbar_desc(pbar, metrics, labels):
    pbar_string = ''
    for metric, label in zip(metrics, labels):
        pbar_string += f'{label}: {metric:.7f}; '
    pbar.set_description(pbar_string)


class Runner():
    def __init__(self, config, logging, local_rank):
        self.config = config
        self.logging = logging
        self.local_rank = local_rank
        self._scaler = GradScaler(enabled=False)

    def train(self):
        train_set, val_set, test_set = get_dataset(self.config)

        models = {
                'cmdNet': cmdNet(iterations=10),
                }

        model = models[self.config.network.model_name].to(self.config.device)
        del models
        if not self.config.ddp:
            model = nn.DataParallel(model)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank)

        if self.config.visdom:
            import visdom
            vis = visdom.Visdom()  # python -m visdom.server
            window = vis.image(np.ones((1, 640, 368 * 5)), opts=dict(title=''))
            # window = vis.image(np.ones((1, 256, 232 * 3)), opts=dict(title=''))

        best_psnr = 0.
        save_results_validation = []
        save_results_test = []
        optimizer = get_optimizer(self.config, model.parameters())
        for epoch in range(1, self.config.training.epoch + 1):
            # -------- training --------
            model.train()
            for step, samples in enumerate(tqdm(train_set)):
                with autocast(enabled=False):
                    atb_tr, org_tr, atb_k_tr, mask_tr, csm_tr = samples  # float
                    org_tr = to_var(org_tr)  # (BS, height, width, complex=2)
                    csm_tr = to_var(csm_tr)  # (BS, coils, height, width, complex=2)
                    mask_tr = to_var(mask_tr)  # (BS, 1, height, width, 1): int
                    atb_tr = to_var(atb_tr)  # (BS, height, width, complex=2)
                    atb_k_tr = to_var(atb_k_tr)  # (BS, coils, height, width, complex=2)
                    optimizer.zero_grad()

                    output, loss_ista = model(atb_tr, atb_k_tr, mask_tr, csm_tr)
                    loss_train = 0.
                    for i in range(1, len(output) + 1):
                        lx = 0.
                        lx += get_loss(output[i-1], org_tr, criterion='mse')
                        lx += get_loss(output[i-1], org_tr, criterion='ssim')
                        times = 10. ** ((i-len(output)) / (len(output)-1))
                        lx *= times
                        lx += torch.mean(torch.pow(loss_ista[i-1], 2)) * 0.01
                        loss_train += lx
                    self._scaler.scale(loss_train).backward()
                    self._scaler.step(optimizer)
                    self._scaler.update()

            # -------- validation --------
            with torch.no_grad():
                model.eval()
                all_psnr, all_ssim, all_nmse = [], [], []
                for step, samples in enumerate(val_set):
                    atb_ts, org_ts, atb_k_ts, mask_ts, csm_ts = samples
                    org_ts = to_var(org_ts)  # (1, 640, 368, 2)
                    csm_ts = to_var(csm_ts)  # (1, 15, 640, 368, 2)
                    mask_ts = to_var(mask_ts)  # (1, 1, 640, 368, 1)
                    atb_ts = to_var(atb_ts)  # (1, 640, 368, 2)
                    atb_k_ts = to_var(atb_k_ts)  # (1, 15, 640, 368, 2)

                    output, _, = model(atb_ts, atb_k_ts, mask_ts, csm_ts)
                    output = output[-1]

                    norm_org = complex_abs(org_ts).data.to('cpu').numpy().squeeze()
                    norm_rec = complex_abs(output).data.to('cpu').numpy().squeeze()
                    all_psnr.append(psnr(norm_org, norm_rec))
                    all_ssim.append(ssim(norm_org, norm_rec))
                    all_nmse.append(nmse(norm_org, norm_rec))

            mean_psnr = np.mean(all_psnr)
            mean_psnr = np.round(mean_psnr, 4)
            std_psnr = np.std(all_psnr)
            std_psnr = np.round(std_psnr, 4)
            mean_ssim = np.mean(all_ssim)
            mean_ssim = np.round(mean_ssim, 4)
            std_ssim = np.std(all_ssim)
            std_ssim = np.round(std_ssim, 4)
            mean_nmse = np.mean(all_nmse)
            mean_nmse = np.round(mean_nmse, 4)
            std_nmse = np.std(all_nmse)
            std_nmse = np.round(std_nmse, 4)

            save_results_validation.append([mean_psnr, mean_ssim, mean_nmse])

            if mean_psnr > best_psnr:
                best_psnr = mean_psnr
                best_model = model
                torch.save(best_model.state_dict(), 'model.pth')
            self.logging.info(f'epoch {epoch}: psnr(val): {mean_psnr}+-{std_psnr}, ssim(val): {mean_ssim}+-{std_ssim}, nmse(val): {mean_nmse}+-{std_nmse}')

