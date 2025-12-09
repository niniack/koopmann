from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)


# WandB
class WandBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    use_wandb: bool
    entity: str | None = None
    project: str | None = None


# Optimizer
class OptimParam(str, Enum):
    adamw = "adamw"
    sgd = "sgd"


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: OptimParam
    weight_decay: NonNegativeFloat
    num_epochs: NonNegativeInt
    learning_rate: PositiveFloat
    batch_size: NonNegativeInt


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
