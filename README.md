# GPU Inference Optimization Lab

Where does LLM inference spend its time on a modern GPU, and how much performance can be recovered through model-, runtime-, memory-, and kernel-level optimization?

This lab prioritizes explainable engineering and reproducible measurements. Phase 0 establishes the environment; Phase 1 provides a trusted baseline. No optimization or speedup is claimed.

## Setup

From this directory, on Linux/WSL with NVIDIA GPU access:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked
./scripts/check_gpu.sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
./scripts/run_baseline.sh --prompt-length 64 --output-length 16 --warmup 1 --runs 3
```

Python 3.12, PyTorch 2.6.0 with bundled CUDA 12.4 runtime, and Transformers 4.51.3 are deliberately pinned. The full dependency resolution is in `uv.lock`; `.venv` is local. The CUDA runtime shipped with PyTorch is different from the maximum CUDA version shown by `nvidia-smi`. Neither a Linux display driver nor a separate CUDA toolkit is required here. `uv` was installed in `~/.local/bin` without modifying shell startup files.

The default is the ungated `HuggingFaceTB/SmolLM2-135M`, a real 135M-parameter base causal LM. FP16 weights are approximately 270 MB before runtime, KV cache, and activations; this is conservative for the detected 8 GB GPU. Base-model completions are not instruction-tuned chat responses. First run downloads weights into the Hugging Face cache. For a different model, inspect its size and available VRAM first; arbitrary model sizes are not automatically admitted or guaranteed to fit.

```bash
uv run --locked python -m benchmarks.baseline --model HuggingFaceTB/SmolLM2-135M \
  --batch-size 1 --prompt-length 512 --output-length 128 --warmup 3 --runs 10
```

Use `--revision <commit>` to repeat a recorded model snapshot. Each run resolves and records the Hub revision. Use the same locked dependencies, workload, dtype, model revision, and hardware conditions for comparisons.

## Layout and measurements

`src/gpu_inference_lab` contains environment inspection, model loading, generation, and metrics. `benchmarks/baseline.py` coordinates them. Tests use a tiny random CPU model, without model downloads, to compare cached greedy generation against a full-prefix reference. Scripts wrap the CLI. JSON results live under ignored `results/raw/`; model weights stay outside the repository.

Timing covers the synchronous wall-clock duration of `model.generate`, including Python orchestration, prefill, and decode. It excludes model loading, prompt tokenization, host-to-device input transfer, and output decoding. It is neither pure GPU kernel time nor full user-request latency. We report repeated latency samples, median, nearest-rank p95, aggregate generated-token throughput, and peak PyTorch allocated/reserved memory. Small-run p95 is just a smoke-test statistic. TTFT and TPOT are null until separately instrumented.

Prompts repeat fixed prose token IDs to an exact length, and greedy decoding disables EOS stopping to enforce equal output work. This is a synthetic performance workload, not a model-quality evaluation. See [methodology](docs/METHODOLOGY.md).

No torch.compile, quantization, profiling traces, vLLM, custom Triton/CUDA kernels, dashboards, or multi-GPU work is implemented. PyTorch's dependency tree includes Triton; no custom Triton work is used. Follow the [roadmap](docs/ROADMAP.md) only after passing each phase's gates.

Read `docs/ENVIRONMENT.md` for local validation results, `docs/DECISIONS.md` for tradeoffs, and `docs/LEARNING_LOG.md` for concepts to explain yourself. The lab files live directly at the repository root, alongside LICENSE. Run setup and benchmark commands from this directory.
