from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

from koopmann.data import DatasetConfig
from scripts.common_config_def import OptimConfig, OptimParam, WandBConfig


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    residual: bool
    out_features: PositiveInt
    hidden_neurons: list[PositiveInt]
    bias: bool
    batchnorm: bool


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    train_data: DatasetConfig
    optim: OptimConfig
    model: ModelConfig
    wandb: WandBConfig
    seed: NonNegativeInt = 0
    print_freq: PositiveInt
    batch_size: PositiveInt
    task_type: str
    save_name: Optional[str] = None
    save_dir: Optional[str] = None
