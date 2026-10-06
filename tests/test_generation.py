import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gpu_inference_lab.generation import generate, prompt_ids


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


def test_prompt_has_exact_length_and_identical_rows():
    actual = prompt_ids(Tokenizer(), 5, 2)
    assert actual.tolist() == [[1, 2, 3, 1, 2], [1, 2, 3, 1, 2]]


def test_greedy_generation_matches_reference_even_with_eos_default():
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            eos_token_id=0,
        )
    ).eval()
    ids = torch.tensor([[1, 2, 3]])
    expected = ids.clone()
    with torch.inference_mode():
        for _ in range(4):
            token = model(expected).logits[:, -1].argmax(dim=-1, keepdim=True)
            expected = torch.cat([expected, token], dim=1)
    actual = generate(model, ids, torch.ones_like(ids), 4, 0)
    assert torch.equal(actual, expected)
    assert actual.shape == (1, 7)


def test_fixed_work_does_not_stop_when_model_predicts_eos():
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            eos_token_id=0,
        )
    ).eval()
    with torch.no_grad():
        model.lm_head.weight.zero_()
    ids = torch.tensor([[1, 2, 3]])
    actual = generate(model, ids, torch.ones_like(ids), 4, 0)
    assert actual.tolist() == [[1, 2, 3, 0, 0, 0, 0]]
