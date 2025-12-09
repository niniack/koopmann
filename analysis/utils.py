from koopmann.models import (
    DecompExponentialKoopmanAutencoder,
    ParamExponentialKoopmanAutencoder,
)
import yaml


def load_autoencoder(file_dir: str, ae_name: str):
    # Autoencoder path in work dir
    ae_file_path = f"{file_dir}/{ae_name}.safetensors"
    autoencoder, ae_metadata = ParamExponentialKoopmanAutencoder.load_model(
        file_path=ae_file_path
    )

    _ = autoencoder.eval()

    return autoencoder, ae_metadata


def load_prompt(path, n):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if "prompts" not in data or not isinstance(data["prompts"], list):
        raise KeyError("YAML file must contain a top-level 'prompts' list.")

    if n < 0 or n > len(data["prompts"]):
        raise IndexError(f"n={n} is out of range (1 to {len(data['prompts'])}).")

    return data["prompts"][n]
