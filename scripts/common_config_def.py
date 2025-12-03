from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
)


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
