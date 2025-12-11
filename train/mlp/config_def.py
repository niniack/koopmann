"""
Pydantic configurations for training ResMLPs.
"""

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)

from koopmann.data import DatasetConfig
from train.common_config_def import EnvSettings, OptimConfig, WandBConfig


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hidden_neurons: list[PositiveInt] | None = None
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
    save_name: str | None = None
    save_dir: str | None = None
    verbose: bool = False

    @model_validator(mode="after")
    def check_wandb_requirements(self):
        wandb = self.wandb
        if wandb and wandb.use_wandb:
            env = EnvSettings()  # type: ignore
            missing = [
                name
                for name in ["WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"]
                if not getattr(env, name)
            ]
            if missing:
                raise ValueError(
                    f"WandB is enabled but missing required environment variables: {', '.join(missing)}"
                )
        return self
