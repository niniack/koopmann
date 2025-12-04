from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, NonNegativeFloat, NonNegativeInt, PositiveInt

from scripts.common_config_def import OptimConfig, WandBConfig


class HostModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hf_name: str


class KoopmanParam(str, Enum):
    exponential = "exponential"


# Autoencoder configuration
class AutoencoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    k_steps: PositiveInt  # Observable space steps
    in_features: PositiveInt  # Input dimension
    ae_dim: PositiveInt  # Observable dimension
    hidden_config: List[PositiveInt]  # Encoder/decoder arch.
    lambda_reconstruction: NonNegativeFloat
    lambda_obs_pred: NonNegativeFloat
    lambda_state_pred: NonNegativeFloat
    bias: bool = True
    batchnorm: bool = False
    koopman_param: Optional[KoopmanParam] = None
    ae_nonlinearity: Optional[str] = None


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
