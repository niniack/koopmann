from pathlib import Path

import fire
import yaml

import wandb
from scripts.train_ae.scale import main as run_scale


def main(sweep_config_path: Path):
    sweep_config = yaml.safe_load(Path(sweep_config_path).read_text())

    # Initialize sweep by passing in config.
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity=sweep_config["entity"],
        project=sweep_config["project"],
    )

    # Start sweep job.
    if "scale" in str(sweep_config_path):
        main_func = run_scale
    else:
        raise ValueError("To be fixed! For now, please send a config file with 'scale'.")
    wandb.agent(sweep_id, function=main_func, count=sweep_config["num_sweeps"] or 5)


if __name__ == "__main__":
    fire.Fire(main)
