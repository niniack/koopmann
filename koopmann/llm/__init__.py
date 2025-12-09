from .data import *
from .model import *

HF_LLM_DICT = {
    # Gemma
    "gemma_3_270m": "google/gemma-3-270m",
    "gemma_3_1b": "google/gemma-3-1b-pt",
    "gemma_3_1b_it": "google/gemma-3-1b-it",
    "gemma_2_2b": "google/gemma-2-2b",
    "gemma_2b": "google/gemma-2b",
    "gemma_2_9b": "google/gemma-2-9b",
    "gemma_7b": "google/gemma-7b",
    # Llama 3.x
    "llama_3p2_1b": "meta-llama/Llama-3.2-1B",
    "llama_3_8b": "meta-llama/Llama-3-8B",
    "llama_3p1_8b": "meta-llama/Llama-3.1-8B",
    # HF SmolLM
    "smollm_135m": "HuggingFaceTB/smollm-135m",
    "smollm_360m": "HuggingFaceTB/smollm-360m",
    "smollm_1p7b": "HuggingFaceTB/smollm-1.7b",
    "smollm_2p7b": "HuggingFaceTB/smollm-2.7b",
    "smollm_8b": "HuggingFaceTB/smollm-8b",
}
