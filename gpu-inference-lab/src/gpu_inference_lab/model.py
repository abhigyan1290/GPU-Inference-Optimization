"""Explicit conservative model default; no remote Python code."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


def load_model(name: str, revision: str, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=False)
    model = (
        AutoModelForCausalLM.from_pretrained(
            name,
            revision=revision,
            torch_dtype=dtype,
            trust_remote_code=False,
            use_safetensors=True,
            attn_implementation="eager",
        )
        .to("cuda")
        .eval()
    )
    return model, tokenizer
