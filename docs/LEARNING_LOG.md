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

- Special tokens: dedicated token IDs can mark sequence start/end, padding, or chat boundaries. Our synthetic prompt currently uses add_special_tokens=False. Check actual tokenizer output before changing this: adding a BOS token changes the measured input and its length. Separate natural-text smoke checks from the fixed-work benchmark.
- Correctness next steps (not yet implemented): use a small Gemma configuration to compare cache-enabled decoding against full-prefix recomputation; compare finite logits with explicit dtype-aware tolerances and greedy tokens; include cache positions near a reduced sliding-window boundary. Add an optional real-model BF16 GPU comparison. A token mismatch near tied logits requires investigation, not silently relaxing the test.
- Repeatability next steps (not yet run): keep one exact workload and model revision fixed; commit the source; use consistent AC power, power mode, and background load; record clocks, power, temperature, and memory. Begin with 5 warm-ups and 20 measurements in each of 3 separate invocations, inspect drift and variability, and adjust based on evidence. More samples do not remove thermal throttling or systematic differences.
