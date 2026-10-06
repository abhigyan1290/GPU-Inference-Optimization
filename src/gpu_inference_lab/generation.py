"""Fixed synthetic workload: repeated prose token IDs and greedy decoding."""

import torch
from transformers import GenerationConfig

PROMPT = "GPU inference involves processing a prompt and then generating tokens one at a time. "


def prompt_ids(tokenizer, length: int, batch_size: int) -> torch.Tensor:
    if length <= 0 or batch_size <= 0:
        raise ValueError("Prompt length and batch size must be positive")
    tokens = tokenizer.encode(PROMPT, add_special_tokens=False)
    if not tokens:
        raise ValueError("Tokenizer produced an empty prompt")
    row = (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]
    return torch.tensor([row] * batch_size, dtype=torch.long)


@torch.inference_mode()
def generate(model, inputs, mask, output_length: int, pad_token_id: int):
    # Fresh config avoids inheriting model-specific sampling or EOS stopping defaults.
    config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=output_length,
        min_new_tokens=output_length,
        eos_token_id=None,
        pad_token_id=pad_token_id,
        use_cache=True,
    )
    # Explicit kwargs apply after Transformers restores model-specific special tokens.
    return model.generate(
        input_ids=inputs,
        attention_mask=mask,
        generation_config=config,
        use_model_defaults=False,
        eos_token_id=None,
    )
