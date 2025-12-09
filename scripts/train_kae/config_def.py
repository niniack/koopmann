from typing import List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
)

from scripts.common_config_def import OptimConfig, WandBConfig, AutoencoderConfig


class HostModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hf_name: str
    layer_start: NonNegativeInt
    layer_end: NonNegativeInt


# Main Config class
class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    optim: OptimConfig
    wandb: WandBConfig
    host_model: HostModelConfig
    autoencoder: AutoencoderConfig
    print_freq: PositiveInt
    verbose: bool
    seed: NonNegativeInt = 0
    save_name: Optional[str] = None
    save_dir: Optional[str] = None
