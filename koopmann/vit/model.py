import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModel


def get_hf_vision_model(
    hf_name: str,
    cache_dir: str,
    device: str | torch.device,
    untrained: bool = False,
):
    """
    Load a vision backbone from Hugging Face,
    plus the corresponding image processor.
    """
    device = torch.device(device)

    if untrained:
        config = AutoConfig.from_pretrained(hf_name, cache_dir=cache_dir)
        model = AutoModel.from_config(config)
    else:
        model = AutoModel.from_pretrained(hf_name, cache_dir=cache_dir)

    model = model.to(device)
    processor = AutoImageProcessor.from_pretrained(hf_name, cache_dir=cache_dir)

    return model, processor
