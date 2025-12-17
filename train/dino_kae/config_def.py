from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)

from train.common_config_def import EnvSettings, OptimConfig, WandBConfig


# Host HF model (DINO, ViT, etc.) configuration
class HostModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hf_name: str
    token_idx: NonNegativeInt
    layer_start: NonNegativeInt
    layer_end: NonNegativeInt


# Autoencoder parameterization
class KoopmanParam(str, Enum):
    vanilla = "vanilla"
    exponential = "exponential"


# Autoencoder configuration
class AutoencoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    k_steps: PositiveInt
    in_features: PositiveInt
    latent_features: PositiveInt
    hidden_config: list[PositiveInt]
    lambda_reconstruction: NonNegativeFloat
    lambda_obs_pred: NonNegativeFloat
    lambda_state_pred: NonNegativeFloat
    koopman_param: KoopmanParam
    bias: bool = True
    batchnorm: bool = False
    nonlinearity: str | None = None


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)
    dataset_name: str
    num_samples: int
    split: str
    seed: int | None = 42


# Main Config class
class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    host_model: HostModelConfig
    train_data: DatasetConfig
    optim: OptimConfig
    autoencoder: AutoencoderConfig
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
