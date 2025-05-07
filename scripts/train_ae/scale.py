import os
from pathlib import Path
from typing import Optional, Union

import fire
import torch
from config_def import Config, KoopmanParam
from safetensors.torch import save_file

import wandb
from koopmann import models as kmodels
from koopmann.log import logger
from koopmann.mixins import Serializable
from koopmann.models import (
    ExponentialKoopmanAutencoder,
    KoopmanAutoencoder,
    LowRankKoopmanAutoencoder,
)
from koopmann.utils import get_device
from scripts.train_ae.losses import AutoencoderMetrics
from scripts.train_ae.shape_metrics import prepare_acts
from scripts.utils import (
    get_dataloaders,
    get_lr_schedule,
    get_optimizer,
    iterate_by_batches,
    setup_config,
)

torch.set_printoptions(precision=4)

############# UTILS #############


def get_model(config, device):
    is_probed = True if config.scale.model_with_probe else False
    if is_probed:
        raise NotImplementedError()

    # Get file path
    file_path = config.scale.model_with_probe if is_probed else config.scale.model_to_scale

    # Get model class
    model_metadata = Serializable.parse_safetensors_metadata(file_path)
    ModelClass = getattr(kmodels, model_metadata["model_class"])

    # Load model
    model, _ = ModelClass.load_model(file_path=file_path)
    model.to(device).eval()

    return model, is_probed


def get_autoencoder(config, model, device):
    autoencoder_kwargs = {
        "k_steps": config.scale.num_scaled_layers,
        # "in_features": model.components[config.scale.scale_location].in_channels,
        "in_features": config.autoencoder.pca_dim
        if config.autoencoder.pca_dim
        else model.components[config.scale.scale_location].in_channels,
        "latent_features": config.autoencoder.ae_dim,
        "hidden_config": config.autoencoder.hidden_config,
        "batchnorm": config.autoencoder.batchnorm,
        "bias": True,
        "rank": config.autoencoder.koopman_rank,
        "nonlinearity": config.autoencoder.ae_nonlinearity,
        "use_eigeninit": False,
    }

    if config.autoencoder.koopman_param == KoopmanParam.exponential:
        autoencoder = ExponentialKoopmanAutencoder(**autoencoder_kwargs).to(device)
        flavor = config.autoencoder.koopman_param.value
    elif config.autoencoder.koopman_param == KoopmanParam.lowrank:
        autoencoder = LowRankKoopmanAutoencoder(**autoencoder_kwargs).to(device)
        flavor = f"{config.autoencoder.koopman_param.value}_{config.autoencoder.koopman_rank}"
    else:
        autoencoder = KoopmanAutoencoder(**autoencoder_kwargs).to(device)
        flavor = "standard"

    return autoencoder, flavor


def save_autoencoder(autoencoder, config, flavor, **kwargs):
    if not config.save_dir:
        return None

    os.makedirs(os.path.dirname(config.save_dir), exist_ok=True)
    filename = (
        f"dim_{config.autoencoder.ae_dim}_"
        f"k_{config.scale.num_scaled_layers}_"
        f"loc_{config.scale.scale_location}_"
        f"{flavor}_"
        f"autoencoder_{config.save_name}.safetensors"
    )
    ae_path = Path(config.save_dir, filename)

    suffix = config.suffix if config.suffix else ""
    suffix = suffix + "_adv" if config.adv.use_adversarial_training else suffix
    autoencoder.save_model(ae_path, suffix=None, **kwargs)

    return ae_path


def check_gradient_norms(model, optimizer, losses):
    grad_norms = {}

    for loss_name, loss_component in losses.items():
        # Clear previous gradients
        optimizer.zero_grad()

        # Backward pass for just this component
        loss_component.backward(retain_graph=True)

        # Calculate gradient norm
        total_norm = (
            sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None)
            ** 0.5
        )

        grad_norms[loss_name] = total_norm

    return grad_norms


########## COMPUTATION ##########


def eval_log_autoencoder(model, autoencoder, act_dict, device, config, epoch):
    # Initialize metrics tracker
    metrics = AutoencoderMetrics(device=device)

    # Set up models
    model.eval().hook_model()
    autoencoder.eval()

    # Forward
    with torch.no_grad():
        batch_size = config.batch_size
        for batch_dict in iterate_by_batches(act_dict, batch_size):
            # Compute losses
            metrics.update(
                autoencoder=autoencoder,
                act_dict=batch_dict,
                k_steps=config.scale.num_scaled_layers,
            )

    # Log metrics
    metrics.log_metrics(epoch, "eval")

    return metrics.compute()


def train_one_epoch(model, autoencoder, act_dict, device, config, epoch, optimizer):
    # Initialize metrics tracker
    metrics = AutoencoderMetrics(device=device)

    # Set up models
    model.to(device).eval().hook_model()
    autoencoder.to(device).train()

    batch_size = config.batch_size
    global_step = epoch * (len(act_dict[0]) // batch_size)
    for batch_dict in iterate_by_batches(act_dict, batch_size):
        # Compute losses

        metrics.update(
            autoencoder=autoencoder,
            act_dict=batch_dict,
            k_steps=config.scale.num_scaled_layers,
        )

        # Loss weighting
        lambda_reconstruction = config.autoencoder.lambda_reconstruction
        lambda_state_pred = config.autoencoder.lambda_state_pred
        lambda_latent_pred = config.autoencoder.lambda_latent_pred
        lambda_isometric = config.autoencoder.lambda_isometric

        losses = {
            "reconstruction": lambda_reconstruction * metrics.batch_metrics.raw_reconstruction,
            "state_pred": lambda_state_pred * metrics.batch_metrics.raw_state_pred,
            "latent_pred": lambda_latent_pred * metrics.batch_metrics.raw_latent_pred,
            "isometric": lambda_isometric * metrics.batch_metrics.raw_distance,
            "sparsity": 0 * metrics.batch_metrics.raw_sparsity,
        }

        # Track the combined loss as before
        loss = sum(losses.values())
        metrics.set_weighted_loss(loss)

        # # Diagnose gradients (only occasionally to avoid slowing training)
        # if global_step % 100 == 0:
        #     grad_norms = check_gradient_norms(autoencoder, optimizer, losses)
        #     wandb.log(grad_norms, step=global_step)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=0.1)
        optimizer.step()
        global_step += 1

    # Calculate averages
    avg_metrics = metrics.compute()

    # Log
    if (epoch + 1) % config.print_freq == 0:
        metrics.log_metrics(epoch, "train")

    # Return results
    return avg_metrics


def main(config_path_or_obj: Optional[Union[Path, str, Config]] = None):
    """Main function to train the autoencoder."""

    # Setup
    config = setup_config(config_path_or_obj, Config)
    device = get_device()

    # Get data
    # `train_subset` and `shuffle` parameters for debugging
    data_train_loader, data_test_loader, train_dataset, test_dataset = get_dataloaders(
        config=config,
        test_batch_size=4096,
        shuffle=True,
        # train_subset=5_000,
        # test_subset=1_000,
    )

    # Load model and create autoencoder
    model, is_probed = get_model(config=config, device=device)
    model.eval()
    autoencoder, flavor = get_autoencoder(config=config, model=model, device=device)
    autoencoder.summary()

    # Setup training
    optimizer = get_optimizer(config, autoencoder)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.optim.num_epochs)
    scheduler = get_lr_schedule(
        lr_schedule_type="cyclic",
        n_epochs=config.optim.num_epochs,
        lr_max=config.optim.learning_rate,
        optimizer=optimizer,
    )

    # Preprocess activations
    train_orig_act_dict, train_proc_act_dict, preproc_dict = prepare_acts(
        data_train_loader=data_train_loader,
        model=model,
        device=device,
        svd_dim=config.autoencoder.pca_dim,
        whiten_alpha=config.autoencoder.whiten_alpha,
        preprocess=config.autoencoder.preprocess,
        only_first_last=True,
    )
    train_act_dict = train_proc_act_dict if config.autoencoder.preprocess else train_orig_act_dict

    metrics = {}
    # Training loop
    for epoch in range(config.optim.num_epochs):
        if config.adv.use_adversarial_training:
            raise NotImplementedError()
        else:
            metrics = train_one_epoch(
                model=model,
                autoencoder=autoencoder,
                act_dict=train_act_dict,
                device=device,
                config=config,
                epoch=epoch,
                optimizer=optimizer,
            )

        scheduler.step()

        # Evaluate
        if (epoch + 1) % config.print_freq == 0:
            eval_log_autoencoder(
                model=model,
                autoencoder=autoencoder,
                act_dict=train_act_dict,
                device=device,
                config=config,
                epoch=epoch,
            )

            logger.info(
                f"Epoch {epoch + 1}/{config.optim.num_epochs}, "
                f"Eval FVU State Pred: {metrics['fvu_state_pred']:.4f}, "
                f"Raw Sparsity: {metrics['raw_sparsity']:.4f}, "
            )

    wandb.finish()

    # Save model
    extra_metadata = {"preprocess": config.autoencoder.preprocess}
    ae_file_name = save_autoencoder(autoencoder, config, flavor, **extra_metadata)

    # Save preprocessing tensors
    save_file(preproc_dict, f"{ae_file_name.parent}/{ae_file_name.stem}_preprocessing.safetensors")


if __name__ == "__main__":
    fire.Fire(main)
