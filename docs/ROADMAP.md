# Roadmap and gates

| Phase | Work | Gate before advancing |
|---|---|---|
| 0 | WSL / CUDA / Python environment | Real CUDA arithmetic succeeds; environment and lock recorded |
| 1 | Trusted PyTorch/Hugging Face baseline | CPU correctness tests pass; real GPU generation and JSON inspected; methodology understood |
| 2 | Workload characterization | Controlled prompt/output/batch sweeps with repeatability and memory limits |
| 3 | PyTorch profiler, Nsight Systems/Compute | Traces explain observed latency; instrumentation overhead assessed |
| 4 | Compile, precision, attention, quantization | Separate correctness checks and equal-work measured comparisons |
| 5 | vLLM/runtime analysis in separate environment | Compatible binary stack; comparable serving semantics and load |
| 6 | Triton kernels | Reference comparisons pass; kernel bottleneck established |
| 7 | Kernel fusion | Measured traffic/launch benefit with correctness retained |
| 8 | Selected CUDA kernels | Explainable implementation, robust reference tests, profiler evidence |
| 9 | Optional multi-GPU | Single-device limits established; communication costs quantified |
| 10 | Potential upstream contribution | Minimal reproducible issue, tests, measured value, project alignment |

Current work stops at Phase 1. Understanding the results is part of the gate, not just passing commands.
