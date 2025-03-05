from pathlib import Path

import fire
import wandb
import yaml

from scripts.train_mlp.train import main as train_func


def main(sweep_config_path: Path):
    sweep_config = yaml.safe_load(Path(sweep_config_path).read_text())

    # Initialize sweep by passing in config.
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity=sweep_config["wandb"]["entity"],
        project=sweep_config["wandb"]["project"],
    )

    # Start sweep job.
    if "train" in str(sweep_config_path):
        main_func = train_func
    else:
        raise ValueError("To be fixed! For now, please send a config file with 'scale' or 'train'.")
    wandb.agent(sweep_id, function=main_func)


if __name__ == "__main__":
    fire.Fire(main)
