import os

from comet_ml import Experiment

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from catalyst.utils import get_device
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback

from utils.dataset import CustomDataset

from utils.augmentation import get_validation_augmentation, get_training_augmentation
from utils.losses import WeightedBCEDiceLoss
from utils.callbacks import CometCallback

from models.EffUNet import EffUNet


os.environ["CUDA_VISIBLE_DEVICES"] = ...
device = get_device()

hyper_params = {
    "in_channels": ...,
    "num_classes": ...,
    "batch_size": ...,
    "num_epochs": ...,
    "learning_rate": 1e-3,
    "lambda_dice": 0.5,
    "lambda_bceWithLogits": 1.5,
    "logdir": ...
}

experiment = Experiment(...)

experiment.log_parameters(hyper_params)

model = EffUNet(in_channels=hyper_params['in_channels'], classes=hyper_params['num_classes'])


images_train, images_valid, masks_train, masks_valid = ..., ..., ..., ...

train_dataset = CustomDataset(images_train, masks_train,
                           transforms=get_training_augmentation())

valid_dataset = CustomDataset(images_valid, masks_valid,
                           transforms=get_validation_augmentation())

train_loader = DataLoader(train_dataset, batch_size=hyper_params['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=hyper_params['batch_size'], shuffle=False)

loaders = {"train": train_loader, "valid": valid_loader}


optimizer = torch.optim.Adam(model.parameters(), hyper_params['learning_rate'])

scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)

criterion = WeightedBCEDiceLoss(
    lambda_dice=hyper_params['lambda_dice'],
    lambda_bce=hyper_params['lambda_bceWithLogits']
)

runner = SupervisedRunner(device=device)

logdir = hyper_params['logdir']

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=[DiceCallback(), CometCallback(experiment), EarlyStoppingCallback(patience=5, min_delta=0.001)],
    logdir=logdir,
    #resume=f"{logdir}/checkpoints/last_full.pth",
    num_epochs=hyper_params['num_epochs'],
    verbose=True
)
