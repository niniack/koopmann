from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from config_def import Config
from torch.utils.data import DataLoader

import wandb
from koopmann.utils import get_device

# from scripts import adv_attacks
from scripts.train_ae.losses import (
    compute_k_prediction_loss,
    compute_latent_space_prediction_loss,
    compute_state_space_recons_loss,
)

# def run_adv_attacks(data_path: str, model_path: str, ae_path: str):
#     return adv_attacks.main.callback(data_path, model_path, ae_path)
