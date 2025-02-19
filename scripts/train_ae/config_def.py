from typing import List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

from koopmann.data import DatasetConfig


class WandBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    use_wandb: bool
    entity: str | None = None
    project: str | None = None


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    num_epochs: PositiveInt
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
class AutoencoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ae_dim: PositiveInt
    lambda_reconstruction: NonNegativeFloat
    lambda_prediction: NonNegativeFloat
    lambda_id: NonNegativeFloat
    exp_param: bool
    batchnorm: bool
    hidden_config: List[PositiveInt]
    ae_nonlinearity: Optional[str] = None


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
