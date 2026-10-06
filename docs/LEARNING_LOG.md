# Engineering learning notebook

Keep entries short: observation, explanation in your own words, evidence, next question.

- CUDA asynchrony: Python returns before queued GPU work completes. Explain why synchronization belongs on both sides of the timed operation.
- Warm-up: initial execution can initialize libraries, memory pools, and caches. TODO: explain what three warm-ups can and cannot guarantee.
- Prefill versus decode: prefill consumes the input context; autoregressive decode adds successive tokens using cached keys/values. TODO: relate each stage to compute and memory traffic.
- TTFT versus TPOT: first-token latency and later per-token latency need separate boundaries. Our total generate timer cannot recover them.
- Throughput: this lab divides batch output tokens by full generation seconds. It includes prefill cost and differs from serving throughput under concurrent arrivals.
- GPU memory: weights, KV cache, activations, allocator reserves, and context overhead differ. TODO: reconcile allocator statistics with nvidia-smi.
- Loading: downloading/deserializing/transferring weights is startup work. It matters operationally but should not contaminate a steady-state generation measurement.
- Driver versus runtime: Windows exposes GPU access to WSL. PyTorch wheels bring their user-space CUDA runtime; nvidia-smi does not tell you which runtime PyTorch uses.
- Correctness: cached generation is tested against repeated full-prefix inference on a tiny random CPU model. This establishes token semantics but is not an exhaustive GPU numerical-accuracy evaluation.
- Reproducibility: a seed alone is insufficient. Lock software and model snapshots, fix work, and record hardware conditions.
