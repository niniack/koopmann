from enum import Enum
from typing import List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)

from koopmann.data import DatasetConfig
from scripts.common_config_def import OptimConfig, WandBConfig


class AdvConfig(BaseModel):
    use_adversarial_training: bool
    epsilon: Optional[NonNegativeFloat] = None

    @model_validator(mode="after")
    def validate_epsilon(self) -> "AdvConfig":
        if self.use_adversarial_training and self.epsilon is None:
            raise ValueError("epsilon is required when use_adversarial_training is True")
        return self


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
    preprocess: bool
    pca_dim: Optional[PositiveInt] = None  # Make optional with default None
    ae_dim: PositiveInt
    lambda_reconstruction: NonNegativeFloat
    lambda_state_pred: NonNegativeFloat
    lambda_latent_pred: NonNegativeFloat
    lambda_isometric: NonNegativeFloat
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

    @model_validator(mode="after")
    def val_preprocess_pca_dim(self) -> "AutoencoderConfig":
        # Validate that pca_dim is provided when preprocess is True
        if self.preprocess and self.pca_dim is None:
            raise ValueError("`pca_dim` must be provided when `preprocess` is True")
        return self


# Main Config class
class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    train_data: DatasetConfig
    optim: OptimConfig
    scale: ScaleConfig
    wandb: WandBConfig
    autoencoder: AutoencoderConfig
    adv: AdvConfig
    batch_size: PositiveInt
    print_freq: PositiveInt
    seed: NonNegativeInt = 0
    task_type: str
    save_name: Optional[str] = None
    save_dir: Optional[str] = None
    suffix: Optional[str] = None
