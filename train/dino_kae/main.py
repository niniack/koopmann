import os
from functools import partial
from pathlib import Path
from pprint import pformat

import fire
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoModel

from koopmann.data import DatasetConfig, get_dataset_class
from koopmann.log import logger
from koopmann.models import KoopmanAutoencoder
from koopmann.utils import get_device
from koopmann.vit import get_hf_vision_model
from train.common_config_def import EnvSettings
from train.common_utils import (
    get_optimizer,
    save_model,
    setup_config,
)
from train.dino_kae.config_def import Config
from train.dino_kae.losses import (
    compute_latent_prediction_loss,
    compute_state_prediction_loss,
    compute_state_recons_loss,
)
from train.dino_kae.utils import (
    build_hidden_states,
    get_autoencoder,
    hf_processor_collate_fn,
)

load_dotenv(Path(__file__).parent.parent.parent / ".env")
HF_HOME = os.getenv("HF_HOME")
DATASETS_CACHE = os.getenv("DATASETS_CACHE")


# Train one epoch
def train_one_epoch(
    autoencoder: KoopmanAutoencoder,
    hf_model: AutoModel,
    dataloader: torch.utils.data.DataLoader,
    config: Config,
    optimizer,
    device: str | torch.device,
    dtype=torch.float32,
):
    """Train the autoencoder for one epoch."""
    autoencoder.train()

    total_loss = torch.tensor(0.0)

    for x_i, x_j, labels in dataloader:
        x_i, x_j = x_i.to(device), x_j.to(device)

        # Compute loss
        recon_loss = compute_state_recons_loss(autoencoder, x_i, x_j)
        latent_pred_loss = compute_latent_prediction_loss(autoencoder, x_i, x_j)
        state_pred_loss = compute_state_prediction_loss(autoencoder, x_i, x_j)
        loss = (
            config.autoencoder.lambda_reconstruction * recon_loss
            + config.autoencoder.lambda_obs_pred * latent_pred_loss
            + config.autoencoder.lambda_state_pred * state_pred_loss
        )
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().cpu()

    avg_loss = total_loss / len(dataloader)
    return avg_loss.item()


def main(
    config_path_or_obj: str,
    env: EnvSettings = EnvSettings(),
):
    """Main function to train the surrogate model."""

    # region Config
    device = get_device()
    config = setup_config(config_path_or_obj, Config, env)
    if config.verbose:
        logger.info(pformat(config.model_dump()))
    # endregion

    # region Load host model in eval mode
    if HF_HOME is None:
        raise RuntimeError("HF_HOME is not set, cannot load HuggingFace model.")
    hf_model, hf_processor = get_hf_vision_model(
        hf_name=config.host_model.hf_name, cache_dir=HF_HOME, device=device
    )
    hf_model.eval()
    # endregion

    # region Load autoencoder in train mode
    autoencoder = get_autoencoder(config=config, device=device).to(
        device, torch.float32
    )
    autoencoder.train().summary()
    # endregion

    # region Load image dataset
    data_config = DatasetConfig(
        dataset_name=config.train_data.dataset_name,
        num_samples=-1,
        split=config.train_data.split,
        root=DATASETS_CACHE,
    )
    DatasetClass = get_dataset_class(data_config.dataset_name)
    dataset = DatasetClass(config=data_config)
    # endregion

    # region Build image dataloader
    custom_hf_collate = partial(hf_processor_collate_fn, hf_processor=hf_processor)
    image_dataloader = DataLoader(
        dataset=dataset,
        batch_size=2048,
        shuffle=False,
        collate_fn=custom_hf_collate,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    # endregion

    # region Build hidden states dataloader
    hs_dataset = build_hidden_states(
        hf_model=hf_model,
        image_dataloader=image_dataloader,
        config=config,
        device=device,
    )
    torch.cuda.empty_cache()
    hs_dataloader = DataLoader(
        dataset=hs_dataset,
        batch_size=config.optim.batch_size,
        shuffle=True,
    )
    logger.info("Built hidden states dataloader")
    # endregion

    # region Training
    optimizer = get_optimizer(config=config, model=autoencoder)

    # Training loop
    for epoch_idx in range(config.optim.num_epochs):
        loss = train_one_epoch(
            autoencoder=autoencoder,
            hf_model=hf_model,
            dataloader=hs_dataloader,
            config=config,
            optimizer=optimizer,
            device=device,
        )

        if epoch_idx % config.print_freq == 0:
            logger.info(f"Epoch {epoch_idx}: Loss = {loss:.4f}")

    # endregion

    # ae_file_name = save_autoencoder(autoencoder, config)


def fire_main():
    torch.set_printoptions(precision=4)
    fire.Fire(main)


if __name__ == "__main__":
    fire_main()
