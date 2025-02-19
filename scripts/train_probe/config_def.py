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


class ProbeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_to_probe: str


class WandBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    use_wandb: bool
    entity: str | None = None
    project: str | None = None


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    weight_decay: NonNegativeFloat
    num_epochs: PositiveInt | None = None
    learning_rate: PositiveFloat


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    train_data: DatasetConfig
    optim: OptimConfig
    wandb: WandBConfig
    probe: ProbeConfig
    seed: NonNegativeInt = 0
    print_freq: PositiveInt
    batch_size: PositiveInt
    task_type: str
    save_name: Optional[str] = None
    save_dir: Optional[str] = None
