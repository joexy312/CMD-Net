from torch.utils.data import DataLoader
from datasets.dataloader_vsNet import load_traindata_path, MRIDataset
import os


def get_dataset(config):

    data_path = os.path.expanduser('~') + config.data.data_path + config.data.vsNet
    data_list = load_traindata_path(data_path, config.training.debug, config.data.sequence)
    train_set = MRIDataset(
        data_list['train'],
        acceleration=config.training.acceleration,
        center_fraction=config.training.center_fraction
    )
    train_set = DataLoader(
        train_set,
        shuffle=True,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_worker,
        # sampler=DistributedSampler(train_set)
    )
    val_set = MRIDataset(
        data_list['val'],
        acceleration=config.training.acceleration,
        center_fraction=config.training.center_fraction
    )
    val_set = DataLoader(
        val_set,
        shuffle=False,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_worker,
        # sampler=DistributedSampler(val_set)
    )
    test_set = MRIDataset(
        data_list['test'],
        acceleration=config.training.acceleration,
        center_fraction=config.training.center_fraction
    )
    test_set = DataLoader(
        test_set,
        shuffle=False,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_worker,
        # sampler=DistributedSampler(test_set)
    )

    return train_set, val_set, test_set

