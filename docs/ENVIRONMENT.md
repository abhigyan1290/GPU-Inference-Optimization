# Local environment validation

Validated 2026-10-06 UTC from this WSL workspace.

| Component | Observed value |
|---|---|
| OS | Ubuntu 24.04.1 LTS, WSL 2 |
| Kernel | 6.18.40.1-microsoft-standard-WSL2 |
| CPU | AMD Ryzen 9 8945HS, 8 cores / 16 logical CPUs |
| WSL-visible RAM | Approximately 7.4 GiB; 2 GiB swap |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8188 MiB |
| Windows NVIDIA driver | 555.97 |
| nvidia-smi reported CUDA support | 12.5 |
| Python | 3.12.3, existing system interpreter |
| uv | 0.12.23, installed in ~/.local/bin |
| PyTorch | 2.6.0+cu124 |
| PyTorch CUDA runtime | 12.4 |
| Transformers | 4.51.3 |
| Compute capability | 8.9 |

CUDA availability and real GPU arithmetic both passed. The full local inspection is saved in ignored `results/raw/environment.json`. All 12 CPU tests passed, including cached greedy decoding against a full-prefix reference and EOS fixed-length handling. Ruff lint and formatting checks passed. `uv pip check` reports all 45 installed packages compatible.

No Windows/Linux driver changes or standalone CUDA toolkit installation were made. Runtime CUDA packages and Triton are dependencies of the selected PyTorch wheel. vLLM is absent. First setup downloaded approximately 3 GiB of wheel archives; uv's extracted cache uses approximately 5.3 GiB, with the virtual environment generally using hardlinks to the cache on this filesystem.

The environment is local to `.venv`; export `PATH="$HOME/.local/bin:$PATH"` if uv is not on your shell path. This snapshot is not a guarantee about another machine. Re-run `./scripts/check_gpu.sh` after driver, WSL, or dependency changes.

## GPU smoke validation

SmolLM2-135M at revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` successfully generated text in FP16. Workload: batch 1, 64 prompt tokens, 16 output tokens, 1 warm-up and 3 measured runs. Each measured run produced exactly 16 tokens. Raw output: `results/raw/20261006T214725523870Z.json` (ignored by Git).

Median generation latency: 0.4688 s; aggregate generated-token throughput: 33.19 tokens/s. These are local smoke measurements, not a stable performance characterization or optimization claim. TTFT and TPOT remain null. JSON metadata, sample counts, actual output counts, and nonempty decoded text were inspected. No setup blockers remain.
