"""
In the hidden states of a HuggingFace LLM,
the 0-th index corresponds to the embeddings,
it's what enters the 0-th decoder block.
"""

from contextlib import contextmanager

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
    device: str | torch.device = "cuda",
    dtype=torch.bfloat16,
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
        raise ValueError(
            "Quantization + untrained random init is not supported in this helper."
        )

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

        if quantize:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            trust_remote_code=True,
            quantization_config=quantization_config,
            cache_dir=cache_dir,
            device_map=device,
            attn_implementation="eager",
            dtype=dtype,
        )

    return model, tokenizer


def get_tlens_llm(
    hf_name,
    model,
    device,
    use_attn_result=False,
    verbose=False,
):
    """
    Get HF LLM.

    `use_attn_result` will also calculate attention head intermediates,
    but it is expensive.
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
        # Toggle whether to explicitly calculate
        # and expose the result for each attention head.
        tl_model.set_use_attn_result(use_attn_result)
    except Exception as e:
        print("`tl_model` is None!")
        if verbose:
            print(e)

    return tl_model


@contextmanager
def inject_token_state(hf_model, layer_j, token_idx, new_vec):
    """
    Temporarily replace the hidden state of a single token at layer_j
    with `new_vec` during forward passes.

    - layer_j: 1-based "hidden_states index" (i.e. hidden_states[layer_j])
    - token_idx: int, position in the sequence to overwrite
    - new_vec: tensor of shape [d] or [1, d] on the correct device
    """
    # Align with hidden_states indexing:
    # hidden_states[0]: embeddings
    # hidden_states[1]: output of layers[0]
    # ...
    hook_layer_index = layer_j - 1
    layer = hf_model.model.layers[hook_layer_index]

    # Ensure shape [d]
    if new_vec.dim() == 2 and new_vec.size(0) == 1:
        new_vec_flat = new_vec[0]
    else:
        new_vec_flat = new_vec

    def hook_fn(module, inp, out):
        # out can be (hidden_states, *rest) or just hidden_states
        if isinstance(out, tuple):
            hidden, *rest = out
            hidden = hidden.clone()
            hidden[:, token_idx, :] = new_vec_flat
            return (hidden, *rest)
        else:
            hidden = out.clone()
            hidden[:, token_idx, :] = new_vec_flat
            return hidden

    handle = layer.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()
