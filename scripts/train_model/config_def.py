from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    ValidationInfo,
    field_validator,
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


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    conv: bool
    residual: bool | None = None
    hidden_neurons: list[PositiveInt] | None = None
    bias: bool
    batchnorm: bool

    @field_validator("residual", "hidden_neurons")
    @classmethod
    def validate_fields_based_on_conv(cls, value, info: ValidationInfo):
        field_name = info.field_name
        data = info.data

        # If the current field is None and conv is False, raise an error
        if "conv" in data and data["conv"] is False and value is None:
            raise ValueError(f"When conv is False, {field_name} must be provided")

        # If the current field is not None and conv is True, raise a warning
        if "conv" in data and data["conv"] is True and value is not None:
            raise ValueError(
                f"When conv is True, {field_name} should not be provided as it will be ignored"
            )

        return value


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    train_data: DatasetConfig
    optim: OptimConfig
    model: ModelConfig
    wandb: WandBConfig
    adv: AdvConfig
    seed: NonNegativeInt = 0
    print_freq: PositiveInt
    batch_size: PositiveInt
    task_type: str
    save_name: Optional[str] = None
    save_dir: Optional[str] = None
    suffix: Optional[str] = None
