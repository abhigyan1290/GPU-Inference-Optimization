"""Inspect the host and actually execute CUDA work before reporting success."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import torch


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check the Windows NVIDIA driver and WSL GPU access."
        )
    x = torch.ones(16, device="cuda")
    if (x @ x).item() != 16:
        raise RuntimeError("CUDA arithmetic validation failed")
    torch.cuda.synchronize()


def collect_environment() -> dict:
    try:
        smi = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=15, check=False
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        smi = str(exc)
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    data = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu,
        "meminfo": Path("/proc/meminfo").read_text(),
        "nvidia_smi": smi,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "safetensors")
        },
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        require_cuda()
        prop = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        data.update(
            cuda_usable=True,
            gpu=prop.name,
            vram_bytes=total,
            free_vram_bytes=free,
            compute_capability=list(torch.cuda.get_device_capability()),
        )
    except (RuntimeError, AssertionError) as exc:
        data.update(cuda_usable=False, cuda_error=str(exc))
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    data = collect_environment()
    print(json.dumps(data, indent=2))
    if args.require_cuda and not data["cuda_usable"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
