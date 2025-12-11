from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
    NonNegativeFloat,
)
from enum import Enum
from scripts.common_config_def import OptimConfig, WandBConfig


class HostModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hf_name: str
    layer_start: NonNegativeInt
    layer_end: NonNegativeInt


# Autoencoder
class KoopmanParam(str, Enum):
    exponential = "exponential"


# Autoencoder configuration
class AutoencoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    k_steps: PositiveInt  # Observable space steps
    in_features: PositiveInt  # Input dimension
    ae_dim: PositiveInt  # Observable dimension
    hidden_config: list[PositiveInt]  # Encoder/decoder arch.
    lambda_reconstruction: NonNegativeFloat
    lambda_obs_pred: NonNegativeFloat
    lambda_state_pred: NonNegativeFloat
    bias: bool = True
    batchnorm: bool = False
    koopman_param: KoopmanParam | None = None
    ae_nonlinearity: str | None = None


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
    save_name: str | None = None
    save_dir: str | None = None
