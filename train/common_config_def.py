"""
Shared Pydantic configurations across training scripts.
"""

from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
)

from pydantic_settings import BaseSettings, SettingsConfigDict


# Environment variables
class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    DATASETS_CACHE: str
    WEIGHTS_CACHE: str
    HF_HOME: str
    WANDB_API_KEY: str | None = None
    WANDB_ENTITY: str | None = None
    WANDB_PROJECT: str | None = None


# WandB
class WandBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    use_wandb: bool


# Optimizer
class OptimParam(str, Enum):
    adamw = "adamw"
    sgd = "sgd"


# Scheduler
class SchedulerParam(str, Enum):
    cyclic = "cyclic"
    cosine = "cosine"


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: OptimParam
    weight_decay: NonNegativeFloat
    num_epochs: NonNegativeInt
    learning_rate: PositiveFloat
    batch_size: NonNegativeInt
    scheduler: SchedulerParam | None = None


def validate_wandb_requirements(env: EnvSettings, wandb: WandBConfig):
    if wandb.use_wandb:
        missing = [
            name
            for name in ["WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"]
            if not getattr(env, name)
        ]
        if missing:
            raise ValueError(
                f"WandB is enabled but missing required environment variables: {', '.join(missing)}"
            )
