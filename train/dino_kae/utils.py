import os
from pathlib import Path
from typing import TypeVar

import torch
from dotenv import load_dotenv
from pydantic import BaseModel
from torch.utils.data import TensorDataset

from koopmann.models import (
    KoopmanAutoencoder,
    ParamExponentialKoopmanAutencoder,
)
from koopmann.shapes import Processor
from train.dino_kae.config_def import KoopmanParam

load_dotenv(Path(__file__).parent.parent.parent / ".env")
TConfig = TypeVar("TConfig", bound=BaseModel)
WEIGHTS_CACHE = os.getenv("WEIGHTS_CACHE")


def build_hidden_states(hf_model, image_dataloader, config, device="cuda"):
    hf_model.to(device)
    hf_model.eval()

    xs_start, xs_end, ys = [], [], []

    with torch.no_grad():
        for images, labels in image_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            x_i, x_j = get_vit_hidden_states(
                hf_model=hf_model,
                input_batch=images,
                token_idx=config.host_model.token_idx,
                layer_start=config.host_model.layer_start,
                layer_end=config.host_model.layer_end,
            )

            xs_start.append(x_i.cpu())
            xs_end.append(x_j.cpu())
            ys.append(labels.cpu())

            del images, labels, x_i, x_j

    xs_start = torch.cat(xs_start, dim=0)
    xs_end = torch.cat(xs_end, dim=0)
    ys = torch.cat(ys, dim=0)

    if config.host_model.use_processor:
        processor = Processor()
        xs_start, xs_end = processor.fit_transform(xs_start, xs_end)

    hidden_dataset = TensorDataset(xs_start, xs_end, ys)

    return hidden_dataset


def get_vit_hidden_states(hf_model, input_batch, token_idx, layer_start, layer_end):
    with torch.no_grad():
        out = hf_model(pixel_values=input_batch, output_hidden_states=True)
    hs = out.hidden_states
    h_start = hs[layer_start][:, token_idx, :]
    h_end = hs[layer_end][:, token_idx, :]
    return h_start, h_end


def hf_processor_collate_fn(batch, hf_processor):
    imgs, labels = zip(*batch)
    processed = hf_processor(images=list(imgs), return_tensors="pt")
    return processed["pixel_values"], torch.tensor(labels)


def get_autoencoder(config: TConfig, device: str | torch.device) -> KoopmanAutoencoder:
    """Get and load autnoencoders."""
    autoencoder_kwargs = {
        "k_steps": config.autoencoder.k_steps,
        "in_features": config.autoencoder.in_features,
        "latent_features": config.autoencoder.latent_features,
        "hidden_config": config.autoencoder.hidden_config,
        "batchnorm": config.autoencoder.batchnorm,
        "bias": config.autoencoder.bias,
        "nonlinearity": config.autoencoder.nonlinearity,
    }

    if config.autoencoder.koopman_param == KoopmanParam.vanilla:
        autoencoder = KoopmanAutoencoder(**autoencoder_kwargs)
    elif config.autoencoder.koopman_param == KoopmanParam.exponential:
        autoencoder = ParamExponentialKoopmanAutencoder(**autoencoder_kwargs)
    else:
        raise ValueError("Please specify Koopman parameterization")

    return autoencoder.to(device).train()
