from koopmann.models import (
    MLP,
    Autoencoder,
    ExponentialKoopmanAutencoder,
    LowRankKoopmanAutoencoder,
    ResMLP,
)


def load_autoencoder(file_dir: str, ae_name: str):
    # Autoenoder path in work dir
    ae_file_path = f"{file_dir}/{ae_name}.safetensors"

    # Choose model based on flag
    if "standard" in ae_name:
        AutoencoderClass = Autoencoder
    elif "lowrank" in ae_name:
        AutoencoderClass = LowRankKoopmanAutoencoder
    elif "exponential" in ae_name:
        AutoencoderClass = ExponentialKoopmanAutencoder

    autoencoder, ae_metadata = AutoencoderClass.load_model(
        ae_file_path,
        strict=True,
        remove_param=True,
    )
    _ = autoencoder.eval()

    return autoencoder, ae_metadata


def load_mlp(file_dir: str, model_name: str) -> tuple:
    """Hooked and in eval mode."""
    # Original model path
    model_file_path = f"{file_dir}/{model_name}.safetensors"

    if "probed" in model_name:
        model, model_metadata = MLP.load_model(file_path=model_file_path)
        model.modules[-2].remove_nonlinearity()
        model.modules[-3].remove_nonlinearity()
        # model.modules[-3].update_nonlinearity("leakyrelu")
    else:
        if "residual" in model_name:
            model, model_metadata = ResMLP.load_model(file_path=model_file_path)
        else:
            model, model_metadata = MLP.load_model(file_path=model_file_path)

    model.eval().hook_model()

    return model, model_metadata
