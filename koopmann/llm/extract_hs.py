from typing import List, Optional, Tuple, Union

import torch


@torch.no_grad()
def extract_hidden_states_from_hf(
    model,
    tokenizer,
    prompts: Union[str, List[str]],
    device: str = "cuda",
    max_length: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract hidden states for a batch of prompts.
    Returns [(B, P, L+1, D), (B, P)]

    B: Batch size
    P: Prompt size
    L: Number of layers
    D: State dimension
    """
    # Batchify, if single prompt
    if isinstance(prompts, str):
        prompts = [prompts]

    # Tokenize
    tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length" if max_length is not None else True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    # shape: [B, P, D]
    # length L+1, first one is embedding output
    hs_tuple = outputs.hidden_states

    # Stack layers
    # shape: [B, P, L+1, D]
    stacked_hs = torch.stack(hs_tuple, dim=2).contiguous()

    return stacked_hs.detach().cpu(), attention_mask.detach().cpu()
