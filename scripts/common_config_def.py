from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)


class OptimParam(str, Enum):
    adamw = "adamw"
    sgd = "sgd"


class WandBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    use_wandb: bool
    entity: str | None = None
    project: str | None = None


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: OptimParam
    weight_decay: NonNegativeFloat
    num_epochs: PositiveInt | None = None
    learning_rate: PositiveFloat
