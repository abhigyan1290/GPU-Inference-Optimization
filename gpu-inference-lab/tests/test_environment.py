import pytest
import torch

from gpu_inference_lab.environment import require_cuda


def test_missing_cuda_is_actionable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Windows NVIDIA driver"):
        require_cuda()
