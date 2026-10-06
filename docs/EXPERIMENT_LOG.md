# Experiment log

A smoke run validates execution, not a performance improvement. Keep exact raw-file references and limitations in every real experiment.

## Template

Experiment:
Date:
Question:
Hypothesis:
Configuration:
Independent variable:
Controlled variables:
Metrics:
Procedure:
Results:
Interpretation:
Limitations:
Next step:

## Phase 0/1 smoke validation

Experiment: GPU baseline harness smoke test
Date: 2026-10-06T21:47:25.523870+00:00
Question: Does the locked environment execute fixed-work GPU generation and save interpretable results?
Hypothesis: The small FP16 baseline fits this GPU and generates exactly the requested tokens.
Configuration: SmolLM2-135M, batch 1, prompt 64, output 16, warm-up 1, runs 3; exact software and revision in `results/raw/20261006T214725523870Z.json`.
Independent variable: None; validation only.
Controlled variables: Model, dtype, input IDs, generation settings, token counts.
Metrics: Synchronized generation wall time, generated tokens, allocator memory.
Procedure: Run environment arithmetic check, CPU correctness suite, then the smoke CLI and inspect JSON.
Results: 12 tests passed; CUDA arithmetic and all three generation runs succeeded with 16 tokens each. Median 0.4688 s; aggregate 33.19 output tokens/s.
Interpretation: The Phase 1 harness works on this machine.
Limitations: Three samples, one warm-up, synthetic repeated prose, uncontrolled laptop clocks; no meaningful tail-latency or speedup claim.
Next step: Developer reviews methodology and learning questions before authorizing Phase 2.
