import os
from pathlib import Path
from pprint import pformat

import fire
import torch
from dotenv import load_dotenv
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from koopmann.llm import (
    LMHiddenStatesDataset,
    extract_hidden_states_from_hf,
    get_hf_llm,
    inject_token_state,
    read_prompts,
)
from koopmann.log import logger
from koopmann.models import KoopmanAutoencoder
from koopmann.utils import get_device
from scripts.train_kae.config_def import Config as ConfigDef
from scripts.train_kae.losses import (
    compute_latent_prediction_loss,
    compute_state_prediction_loss,
    compute_state_recons_loss,
)
from scripts.train_kae.utils import (
    build_autoencoder,
    get_optimizer,
    save_autoencoder,
    setup_config,
)

load_dotenv(Path(__file__).parent.parent.parent / ".env")
HF_HOME = os.getenv("HF_HOME")


def train_one_epoch(
    autoencoder: KoopmanAutoencoder,
    data_loader: torch.utils.data.DataLoader,
    config: ConfigDef,
    optimizer,
    device: str | torch.device,
    dtype=torch.float32,
):
    """Train the autoencoder for one epoch."""
    autoencoder.train()
    total_loss = torch.tensor(0.0)
    for x_i, x_j in data_loader:
        # Data
        x_i = x_i.to(device).to(dtype)
        x_j = x_j.to(device).to(dtype)

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

    avg_loss = total_loss / len(data_loader)
    return avg_loss.item()


def main(config_path_or_obj: str, prompts_path: str):
    """Main function to train the surrogate model."""

    # Config
    device = get_device()
    config = setup_config(config_path_or_obj, ConfigDef)
    if config.verbose:
        logger.info(pformat(config.model_dump()))

    # Load host model
    if HF_HOME is None:
        raise RuntimeError("HF_HOME is not set, cannot load HuggingFace model.")
    hf_model, hf_tokenizer = get_hf_llm(
        hf_name=config.host_model.hf_name,
        cache_dir=HF_HOME,
        device=device,
    )
    hf_model.eval()

    # Load autoencoder
    autoencoder = build_autoencoder(config=config, device=device).to(
        device, torch.float32
    )
    autoencoder.summary()

    # Dataset
    hidden_states, attention_mask = extract_hidden_states_from_hf(
        model=hf_model,
        tokenizer=hf_tokenizer,
        prompts=read_prompts(prompts_path),
        device=device,
    )

    dataset = LMHiddenStatesDataset(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        layer_i=config.host_model.layer_start,
        layer_j=config.host_model.layer_end,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=config.optim.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Setup training
    optimizer = get_optimizer(config=config, model=autoencoder)

    # Training loop
    for epoch_idx in range(config.optim.num_epochs):
        loss = train_one_epoch(
            autoencoder=autoencoder,
            data_loader=data_loader,
            config=config,
            optimizer=optimizer,
            device=device,
        )

        if epoch_idx % config.print_freq == 0:
            logger.info(f"Epoch {epoch_idx}: Loss = {loss:.4f}")

    ae_file_name = save_autoencoder(autoencoder, config)


def fire_main():
    torch.set_printoptions(precision=4)
    fire.Fire(main)


if __name__ == "__main__":
    fire_main()
