import torch
import yaml
from torch.utils.data import Dataset


def read_prompts(filepath: str) -> list[str]:
    with open(filepath) as f:
        data = yaml.safe_load(f)
    return data["prompts"]


class LMHiddenStatesDataset(Dataset):
    """
    Wraps precomputed hidden states into (x_i, x_j) pairs.
    """

    def __init__(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_i: int,
        layer_j: int,
        mode: str = "last",
    ):
        assert hidden_states.ndim == 4, f"Expected 4D tensor, got {hidden_states.shape}"
        assert attention_mask.shape[:2] == hidden_states.shape[:2]

        self.hidden_states = hidden_states
        self.attention_mask = attention_mask
        self.layer_i = layer_i
        self.layer_j = layer_j

        # all (b, t) where mask==1
        if mode == "all":
            b_idx, t_idx = torch.nonzero(attention_mask, as_tuple=True)
            self.index_tuples = list(zip(b_idx.tolist(), t_idx.tolist()))
        elif mode == "last":
            self.index_tuples = []
            for b in range(attention_mask.shape[0]):
                seq_len = attention_mask[b].sum().item()
                self.index_tuples.append((b, seq_len - 1))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __len__(self) -> int:
        return len(self.index_tuples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = self.index_tuples[idx]
        x_i = self.hidden_states[b, t, self.layer_i, :]
        x_j = self.hidden_states[b, t, self.layer_j, :]
        return x_i, x_j


@torch.no_grad()
def extract_hidden_states_from_hf(
    model,
    tokenizer,
    prompts: str | list[str],
    device: str | torch.device = "cuda",
    max_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
