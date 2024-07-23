import os
import logging
import sys
import hydra
import traceback
from runners.runner import Runner

import numpy as np
from utils.util import dict2namespace
import multiprocessing


@hydra.main(config_path='configs', config_name='cmdnet')
def main(config):
    config = dict2namespace(config)
    import torch
    # torch.autograd.set_detect_anomaly(True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.training.gpu_id)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config.device = device

    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
    torch.backends.cudnn.benchmark = True

    logging.info("Using device: {}".format(device))
    logging.info(f'{config}')

    try:
        runner = Runner(config, logging, None)
        runner.train()
    except:
        logging.error(traceback.format_exc())
    return 0


if __name__ == '__main__':
    sys.exit(main())
