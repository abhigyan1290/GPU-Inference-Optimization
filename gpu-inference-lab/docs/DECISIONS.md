# Decisions

## 001 — User-space environment
Accepted: use installed Python 3.12 and uv with a project-local virtual environment and a committed lockfile. Pin PyTorch 2.6.0 CUDA 12.4 to suit observed Windows driver 555.97 (nvidia-smi reports CUDA 12.5). This is a compatibility baseline, not a claim to use the newest stack. Do not install Linux display drivers, vLLM, torchvision, torchaudio, or a standalone toolkit. Future upgrades are explicit experimental changes.

References: https://docs.pytorch.org/get-started/previous-versions/ and https://docs.astral.sh/uv/getting-started/installation/ .

## 002 — Small real model
Accepted: HuggingFaceTB/SmolLM2-135M in FP16; approximately 270 MB of weights leaves substantial headroom on an 8 GB GPU for a small baseline. A larger model might represent other workloads better but adds download and memory costs before the harness is validated. Record and allow an explicit model revision. Prefer safetensors and no remote code.

Model card: https://huggingface.co/HuggingFaceTB/SmolLM2-135M .

## 003 — Honest fixed-work measurements
Accepted: synchronized wall time around generate, eager attention, KV caching, exact synthetic input and output lengths, and no EOS stopping. Report prefill-plus-decode throughput; defer TTFT and TPOT. This sacrifices natural termination and realistic mixed prompts for controlled work. See METHODOLOGY.md before comparing results.

## 004 — Repository boundary
The requested gpu-inference-lab directory is inside the existing workspace. Git discovery resolves to /home/abhigyandoshi. Following the instruction to initialize only if not already inside a repository, no nested Git repository is created and no home-directory files are staged. Commit only the lab files intentionally; environment and raw results are ignored.
