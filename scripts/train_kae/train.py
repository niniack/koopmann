import os
from pathlib import Path
from typing import Optional, Union

import fire
import torch
import wandb
from dotenv import load_dotenv
from utils import build_autoencoder

from koopmann.llm import extract_hidden_states_from_hf, get_hf_llm
from koopmann.utils import get_device
from scripts.train_kae.config_def import Config
from scripts.train_kae.utils import setup_config

load_dotenv("../.env")
HF_HOME = os.getenv("HF_HOME")


def train_one_epoch(model, autoencoder, act_dict, device, config, epoch, optimizer):
    pass


def main(config_path_or_obj: Optional[Union[Path, str, Config]] = None):
    """Main function to train the autoencoder."""

    # Dummy config
    config = setup_config(config_path_or_obj, Config)

    # Setup
    device = get_device()

    # Load model and create autoencoder
    hf_model, hf_tokenizer = get_hf_llm(
        hf_name=config.host_model.hf_name, cache_dir=HF_HOME, device=device
    )
    hf_model.eval()
    autoencoder = build_autoencoder(config=config, device=device)
    autoencoder.summary()

    # # Setup training
    # optimizer = get_optimizer(config, autoencoder)
    # # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.optim.num_epochs)
    # scheduler = get_lr_schedule(
    #     lr_schedule_type="cyclic",
    #     n_epochs=config.optim.num_epochs,
    #     lr_max=config.optim.learning_rate,
    #     optimizer=optimizer,
    # )

    # # Preprocess activations
    # train_orig_act_dict, train_proc_act_dict, preproc_dict = prepare_acts(
    #     data_train_loader=data_train_loader,
    #     model=model,
    #     device=device,
    #     svd_dim=config.autoencoder.pca_dim,
    #     whiten_alpha=config.autoencoder.whiten_alpha,
    #     preprocess=config.autoencoder.preprocess,
    #     only_first_last=True,
    # )
    # train_act_dict = train_proc_act_dict if config.autoencoder.preprocess else train_orig_act_dict

    # metrics = {}
    # # Training loop
    # for epoch in range(config.optim.num_epochs):
    #     metrics = train_one_epoch(
    #         model=model,
    #         autoencoder=autoencoder,
    #         act_dict=train_act_dict,
    #         device=device,
    #         config=config,
    #         epoch=epoch,
    #         optimizer=optimizer,
    #     )

    #     scheduler.step()

    #     # Evaluate
    #     if (epoch + 1) % config.print_freq == 0:
    #         eval_log_autoencoder(
    #             model=model,
    #             autoencoder=autoencoder,
    #             act_dict=train_act_dict,
    #             device=device,
    #             config=config,
    #             epoch=epoch,
    #         )

    #         logger.info(
    #             f"Epoch {epoch + 1}/{config.optim.num_epochs}, "
    #             f"Eval FVU State Pred: {metrics['fvu_state_pred']:.4f}, "
    #         )

    # wandb.finish()


if __name__ == "__main__":
    # For debugging
    torch.set_printoptions(precision=4)
    fire.Fire(main)
