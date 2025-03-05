from enum import Enum
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


class WandBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    use_wandb: bool
    entity: str | None = None
    project: str | None = None


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    residual: bool
    out_features: PositiveInt
    hidden_neurons: list[PositiveInt]


class OptimParam(str, Enum):
    adamw = "adamw"
    sgd = "sgd"


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: OptimParam
    weight_decay: NonNegativeFloat
    num_epochs: PositiveInt | None = None
    learning_rate: PositiveFloat


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
