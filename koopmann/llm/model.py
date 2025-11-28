import torch
from transformer_lens import HookedTransformer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def get_hf_llm(
    hf_name: str,
    cache_dir: str,
    quantize: bool = False,
    untrained: bool = False,
    device: str = "cuda",
    dtype: torch.tensor.dtype = torch.bfloat16,
):
    """
    Get LLM and tokenizer from HF in bf16.
    - If `untrained=False`: load pretrained weights.
    - If `untrained=True`: create a randomly initialized model from config
      (no pretrained weights downloaded).
    - If `quantize=True`: load in 8-bit with bitsandbytes (pretrained only).
    - If `use_bf16=True` and not quantized: load in bfloat16.
    """

    if untrained and quantize:
        raise ValueError("Quantization + untrained random init is not supported in this helper.")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        hf_name,
        use_fast=True,
        cache_dir=cache_dir,
    )

    # Random weights
    if untrained:
        config = AutoConfig.from_pretrained(
            hf_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        model = AutoModelForCausalLM.from_config(config)
        if device is not None:
            model.to(device=device, dtype=dtype)

    else:
        quantization_config = None
        torch_dtype = None

        if quantize:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            torch_dtype = dtype

        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            trust_remote_code=True,
            quantization_config=quantization_config,
            cache_dir=cache_dir,
            device_map=device,
            attn_implementation="eager",
            torch_dtype=torch_dtype,
        )

    return model, tokenizer


def get_tlens_llm(hf_name, model, device, use_attn_result=False, verbose=False):
    """
    Get HF LLM.

    `use_attn_result` will also calculate attention head intermediates, but expensive.
    """
    tl_model = None
    try:
        tl_model = HookedTransformer.from_pretrained(
            hf_name,
            hf_model=model,
            device=device,
            fold_value_biases=False,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
        # Toggle whether to explicitly calculate and expose the result for each attention head.
        tl_model.set_use_attn_result(use_attn_result)
    except Exception as e:
        print("`tl_model` is None!")
        if verbose:
            print(e)

    return tl_model
