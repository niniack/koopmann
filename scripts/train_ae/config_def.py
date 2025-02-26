from enum import Enum
from typing import List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from koopmann.data import DatasetConfig


class WandBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    use_wandb: bool
    entity: str | None = None
    project: str | None = None


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    num_epochs: NonNegativeInt
    learning_rate: PositiveFloat
    weight_decay: NonNegativeFloat
    betas: list[PositiveFloat] | None = None


# ScaleConfig for scale-related parameters
class ScaleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_to_scale: str
    model_with_probe: Optional[str] = None
    scale_location: NonNegativeInt
    num_scaled_layers: PositiveInt


# Autoencoder configuration
class KoopmanParam(str, Enum):
    exponential = "exponential"
    lowrank = "lowrank"


class AutoencoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ae_dim: PositiveInt
    lambda_reconstruction: NonNegativeFloat
    lambda_prediction: NonNegativeFloat
    lambda_id: NonNegativeFloat
    batchnorm: bool
    hidden_config: List[PositiveInt]
    koopman_param: Optional[KoopmanParam] = None
    koopman_rank: Optional[int] = None
    ae_nonlinearity: Optional[str] = None

    @model_validator(mode="after")
    def val_koopman_rank(self) -> "AutoencoderConfig":
        # Only validate koopman_rank when koopman_param is `lowrank`
        if self.koopman_param == KoopmanParam.lowrank:
            if self.koopman_rank is None or self.koopman_rank <= 0:
                raise ValueError(
                    "`koopman_rank` must be a positive integer when `koopman_param` is `lowrank`"
                )
        return self


# Main Config class
class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    train_data: DatasetConfig
    optim: OptimConfig
    scale: ScaleConfig
    wandb: WandBConfig
    autoencoder: AutoencoderConfig
    batch_size: PositiveInt
    print_freq: PositiveInt
    seed: NonNegativeInt = 0
    task_type: str
    save_name: Optional[str] = None
    save_dir: Optional[str] = None
